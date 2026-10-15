"""Tensor serialization helpers for vllm-lens activations and hook results."""

from __future__ import annotations

import base64
import contextlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import cloudpickle
import numpy as np
import torch
import zstandard as zstd

if TYPE_CHECKING:
    from vllm_lens._helpers._activation_store import ActivationStore

_ZSTD_COMPRESSOR = zstd.ZstdCompressor(level=1)
_ZSTD_DECOMPRESSOR = zstd.ZstdDecompressor()

# torch dtypes that numpy cannot represent natively.
_TORCH_TO_NUMPY_VIEW: dict[torch.dtype, np.dtype[Any]] = {
    torch.bfloat16: np.dtype(np.int16),
}


def _encode_tensor(tensor: torch.Tensor) -> tuple[bytes, dict[str, Any]]:
    """Compress a tensor to zstd bytes + the metadata needed to round-trip it.

    Shared by the base64 (:func:`serialize_tensor`) and binary
    (:func:`serialize_tensor_binary`) transports so both encode bit-identically.
    """
    t = tensor.detach().cpu()
    original_dtype = str(t.dtype)

    view_dtype = _TORCH_TO_NUMPY_VIEW.get(t.dtype)
    if view_dtype is not None:
        arr = t.view(torch.int16).numpy()
    else:
        arr = t.numpy()

    compressed = _ZSTD_COMPRESSOR.compress(arr.tobytes())
    meta = {
        "dtype": str(arr.dtype),
        "original_dtype": original_dtype,
        "shape": list(arr.shape),
        "compression": "zstd",
    }
    return compressed, meta


# Every packed tensor starts on this byte boundary, so any dtype view of it is aligned.
_PACK_ALIGN = 64
# Threads for the shared file: pwrite and preadv release the GIL, and one thread is bound by
# its own memory copy, not by the file system.
_IO_THREADS = 8
# Entries at least this large on average go to the shared files straight from their chunks.
_DIRECT_ENTRY_BYTES = 256 * 1024


def _pwrite_all(fd: int, chunks: list[torch.Tensor], start: int) -> None:
    """Write ``chunks`` back to back from byte ``start``; a call writes at most about 2 GB."""
    for chunk in chunks:
        flat = chunk.detach().reshape(-1)
        # A one-element slice counts as contiguous but can keep its parent's stride.
        if flat.stride(-1) != 1:
            flat = flat.clone(memory_format=torch.contiguous_format)
        view = memoryview(flat.view(torch.uint8).numpy())
        while view:
            written = os.pwrite(fd, view, start)
            if written == 0:
                raise OSError(f"the file accepted no bytes at offset {start}")
            view, start = view[written:], start + written


def _pread_all(fd: int, tensor: torch.Tensor, offset: int) -> None:
    """Fill ``tensor`` from byte ``offset`` of ``fd``."""
    view = memoryview(tensor.view(torch.uint8).reshape(-1).numpy())
    while view:
        read = os.preadv(fd, [view], offset)
        if read == 0:
            raise OSError(f"the file ends before offset {offset}")
        view, offset = view[read:], offset + read


def pack_chunks(
    entries: list[tuple[str, str, torch.dtype, list[int], list[torch.Tensor]]],
    path: str | None = None,
) -> dict[str, Any]:
    """Pack ``(id, key, dtype, shape, chunks)`` entries as one uint8 buffer and an index; each
    entry's chunks, flattened in order, fill its tensor. With ``path`` the bytes go to
    ``_IO_THREADS`` shared-memory files ``path.<n>``, written in parallel, and only the paths
    travel: large entries straight from their chunks, small ones as byte ranges of a buffer."""
    sizes = [math.prod(shape) * dtype.itemsize for _, _, dtype, shape, _ in entries]
    if path is not None and sum(sizes) >= _DIRECT_ENTRY_BYTES * max(len(entries), 1):
        return _pack_entries_to_files(entries, sizes, path)
    index: list[tuple[str, str, str, list[int], int, int]] = []
    offset = 0
    for (req_id, key, dtype, shape, _), size in zip(entries, sizes):
        index.append((req_id, key, str(dtype), shape, offset, size))
        offset += -(-size // _PACK_ALIGN) * _PACK_ALIGN
    buffer = torch.empty(offset, dtype=torch.uint8)
    for (_, _, dtype, _, chunks), (*_, start, size) in zip(entries, index):
        target = buffer[start : start + size].view(dtype)
        position = 0
        for chunk in chunks:
            flat = chunk.detach().reshape(-1)
            target[position : position + flat.numel()].copy_(flat)
            position += flat.numel()
    if path is None:
        return {"packed_index": index, "packed_buffer": buffer.numpy()}
    # A few large writes, one file per thread: per-entry calls cost more than the bytes, and
    # writes to one file serialize on its inode lock.
    ranges = _byte_ranges(offset)
    try:
        with ThreadPoolExecutor(len(ranges)) as pool:
            for write in [
                pool.submit(_write_file, f"{path}.{n}", buffer[low:high])
                for n, (low, high) in enumerate(ranges)
            ]:
                write.result()
    except BaseException:
        _remove_shared_files(path)
        raise
    return {"packed_index": index, "packed_path": path, "packed_size": offset}


def _pack_entries_to_files(
    entries: list[tuple[str, str, torch.dtype, list[int], list[torch.Tensor]]],
    sizes: list[int],
    path: str,
) -> dict[str, Any]:
    """Write each entry's chunks straight to one of ``_IO_THREADS`` files, entries dealt round
    the files: for large entries, the copy into a buffer would cost more than the calls."""
    index: list[tuple[str, str, str, list[int], int, int]] = []
    shard_of: list[int] = []
    ends = [0] * _IO_THREADS
    for number, ((req_id, key, dtype, shape, _), size) in enumerate(
        zip(entries, sizes)
    ):
        shard = number % _IO_THREADS
        index.append((req_id, key, str(dtype), shape, ends[shard], size))
        shard_of.append(shard)
        ends[shard] += -(-size // _PACK_ALIGN) * _PACK_ALIGN

    def _write_shard(shard: int) -> None:
        fd = os.open(f"{path}.{shard}", os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
        try:
            for (*_, chunks), (*_, start, _), owner in zip(entries, index, shard_of):
                if owner == shard:
                    _pwrite_all(fd, chunks, start)
        finally:
            os.close(fd)

    try:
        with ThreadPoolExecutor(_IO_THREADS) as pool:
            for write in [
                pool.submit(_write_shard, shard) for shard in range(_IO_THREADS)
            ]:
                write.result()
    except BaseException:
        _remove_shared_files(path)
        raise
    return {"packed_index": index, "packed_path": path, "packed_shards": shard_of}


def _remove_shared_files(path: str) -> None:
    """Remove every shard, including partial files left by a failed transfer."""
    for shard in range(_IO_THREADS):
        with contextlib.suppress(FileNotFoundError):
            os.unlink(f"{path}.{shard}")


def _byte_ranges(size: int) -> list[tuple[int, int]]:
    """``size`` bytes split into ``_IO_THREADS`` ranges, one per shared-memory file."""
    step = -(-size // _IO_THREADS) if size else 0
    return [
        (min(n * step, size), min((n + 1) * step, size)) for n in range(_IO_THREADS)
    ]


def _write_file(path: str, data: torch.Tensor) -> None:
    """Write ``data`` to a new file at ``path``."""
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    try:
        _pwrite_all(fd, [data], 0)
    finally:
        os.close(fd)


def _read_file(path: str, data: torch.Tensor) -> None:
    """Fill ``data`` from the file at ``path``, then remove the file."""
    fd = os.open(path, os.O_RDONLY)
    try:
        _pread_all(fd, data, 0)
    finally:
        os.close(fd)
        os.unlink(path)


def pack_payloads(
    payloads: dict[str, dict[str, dict[str, torch.Tensor]]],
) -> dict[str, Any]:
    """Pack ``{id: {"activations": {key: tensor}}}`` with :func:`pack_chunks`.
    Pickling the array copies one buffer; pickling the tensors saves each storage alone."""
    return pack_chunks(
        [
            (req_id, key, tensor.dtype, list(tensor.shape), [tensor])
            for req_id, payload in payloads.items()
            for key, tensor in payload["activations"].items()
        ]
    )


def unpack_payloads(
    packed: dict[str, Any],
) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    """Rebuild :func:`pack_chunks` input. Each tensor is copied out of the shared buffer, so
    pickling one later does not carry the others; the shared-memory files are removed."""
    path = packed.get("packed_path")
    if path is not None and "packed_shards" in packed:
        return _unpack_entries_from_files(packed)
    if path is None:
        buffer = torch.from_numpy(packed["packed_buffer"])
    else:
        buffer = torch.empty(packed["packed_size"], dtype=torch.uint8)
        ranges = _byte_ranges(packed["packed_size"])
        try:
            with ThreadPoolExecutor(len(ranges)) as pool:
                for read in [
                    pool.submit(_read_file, f"{path}.{n}", buffer[low:high])
                    for n, (low, high) in enumerate(ranges)
                ]:
                    read.result()
        finally:
            _remove_shared_files(path)
    out: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for req_id, key, dtype, shape, offset, size in packed["packed_index"]:
        tensor = (
            buffer[offset : offset + size]
            .view(getattr(torch, dtype.removeprefix("torch.")))
            .reshape(shape)
            .clone()
        )
        out.setdefault(req_id, {"activations": {}})["activations"][key] = tensor
    return out


def _unpack_entries_from_files(
    packed: dict[str, Any],
) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    """Read each entry of :func:`_pack_entries_to_files` straight into its own tensor, then
    remove the files."""
    path = packed["packed_path"]
    fds: list[int] = []
    out: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    try:
        for shard in range(_IO_THREADS):
            fds.append(os.open(f"{path}.{shard}", os.O_RDONLY))
        with ThreadPoolExecutor(_IO_THREADS) as pool:
            reads = []
            entries = zip(packed["packed_index"], packed["packed_shards"])
            for (req_id, key, dtype, shape, offset, _), shard in entries:
                tensor = torch.empty(
                    shape, dtype=getattr(torch, dtype.removeprefix("torch."))
                )
                reads.append(pool.submit(_pread_all, fds[shard], tensor, offset))
                out.setdefault(req_id, {"activations": {}})["activations"][key] = tensor
            for read in reads:
                read.result()
    finally:
        for fd in fds:
            os.close(fd)
        _remove_shared_files(path)
    return out


def serialize_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    """Convert a single torch.Tensor to a JSON-serializable base64 dict.

    Preserves the tensor's native dtype (including bfloat16) and applies
    zstd compression before base64-encoding for ~4-6× size reduction
    compared to the previous float32 + uncompressed path.

    Output::

        {
            "data": "<b64-of-zstd-compressed-bytes>",
            "dtype": "int16",          # numpy storage dtype
            "original_dtype": "torch.bfloat16",  # for faithful round-trip
            "shape": [...],
            "compression": "zstd",
        }
    """
    compressed, meta = _encode_tensor(tensor)
    return {"data": base64.b64encode(compressed).decode("ascii"), **meta}


def deserialize_tensor(d: dict[str, Any]) -> torch.Tensor:
    """Convert a base64 dict back to a torch.Tensor.

    Handles both the new compressed/native-dtype format and the legacy
    uncompressed float32 format for backward compatibility.
    """
    return tensor_from_bytes(base64.b64decode(d["data"]), d)


def tensor_from_bytes(raw: bytes, meta: dict[str, Any]) -> torch.Tensor:
    """Reconstruct a tensor from transport bytes + its metadata.

    ``raw`` is the payload exactly as produced by :func:`_encode_tensor` (a zstd
    frame when ``meta["compression"] == "zstd"``). Shared by the base64 path
    (after ``b64decode``) and the binary-handle path (after fetching the bytes
    from ``GET /v1/activations/{handle}``), so both reconstruct identically.
    """
    if meta.get("compression") == "zstd":
        raw = _ZSTD_DECOMPRESSOR.decompress(raw)

    arr = np.frombuffer(raw, dtype=np.dtype(meta["dtype"])).reshape(meta["shape"])
    t = torch.from_numpy(arr.copy())

    original_dtype = meta.get("original_dtype")
    if original_dtype == "torch.bfloat16":
        t = t.view(torch.bfloat16)
    elif original_dtype == "torch.float16":
        t = t.to(torch.float16)

    return t


def deserialize_directions(value: str | dict[str, Any]) -> torch.Tensor:
    """Decode an ``output_residual_stream_project`` value to float32 directions."""
    return deserialize_tensor(
        json.loads(value) if isinstance(value, str) else value
    ).float()


def serialize_activations(tensor_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert a flat activations dict to a JSON-serializable form.

    Tensors keep the existing wire format (see :func:`serialize_tensor`);
    JSON-safe non-tensor values (e.g. ``qk_layers`` / ``qk_meta`` from
    attention Q/K capture) are wrapped as ``{"__json__": value}``.

    Input::

        {"residual_stream": Tensor(n_layers, total_pos, d), "qk_layers": [2]}

    Output::

        {"residual_stream": {"data": "<b64>", ...}, "qk_layers": {"__json__": [2]}}
    """
    return {
        name: serialize_tensor(t) if isinstance(t, torch.Tensor) else {"__json__": t}
        for name, t in tensor_dict.items()
    }


def decode_activation_entry(encoded: Any) -> Any:
    """Decode one value of a serialized activations dict."""
    if isinstance(encoded, dict) and "__json__" in encoded:
        return encoded["__json__"]
    return deserialize_tensor(encoded)


def serialize_tensor_binary(
    tensor: torch.Tensor, store: ActivationStore
) -> dict[str, Any]:
    """Park a tensor's zstd bytes in ``store``; return a JSON handle descriptor.

    The heavy bytes never enter the JSON body — the client fetches them from
    ``GET /v1/activations/{handle}`` and reconstructs with the metadata here.
    Shape identical to :func:`serialize_tensor` minus ``data``, plus
    ``handle`` / ``nbytes`` / ``transport``.
    """
    compressed, meta = _encode_tensor(tensor)
    handle = store.put(compressed, meta)
    return {"handle": handle, "nbytes": len(compressed), "transport": "binary", **meta}


def serialize_activations_binary(
    tensor_dict: dict[str, Any], store: ActivationStore
) -> dict[str, Any]:
    """Binary-transport counterpart of :func:`serialize_activations`.

    Non-tensor values are wrapped as ``{"__json__": value}`` exactly as in
    the base64 path.  All tensors of one response are stored in a single atomic
    :meth:`ActivationStore.put_many`, so a later tensor's insert can never
    evict an earlier one's handle before the client has fetched it.
    """
    names = [n for n, t in tensor_dict.items() if isinstance(t, torch.Tensor)]
    encoded = [_encode_tensor(tensor_dict[n]) for n in names]
    handles = store.put_many(encoded)
    out: dict[str, Any] = {
        name: {
            "handle": handle,
            "nbytes": len(compressed),
            "transport": "binary",
            **meta,
        }
        for name, handle, (compressed, meta) in zip(names, handles, encoded)
    }
    # Non-tensor entries (qk_layers / qk_meta) ride in the JSON as usual.
    for name, t in tensor_dict.items():
        if not isinstance(t, torch.Tensor):
            out[name] = {"__json__": t}
    return out


def decode_activations(
    response_json: dict[str, Any],
    *,
    fetch_bytes: Callable[[str], bytes] | None = None,
) -> dict[str, Any]:
    """Decode activations from an HTTP API response, base64 or binary.

    Takes the raw JSON response dict from ``/v1/completions`` or
    ``/v1/chat/completions`` and converts any activation tensors back to
    ``torch.Tensor``; non-tensor ``{"__json__": ...}`` entries (e.g. ``qk_layers``
    / ``qk_meta`` from attention Q/K capture) pass through as-is.  Two
    transports are supported per tensor entry:

    - **base64** (default): the entry carries ``data``; decoded in-process.
    - **binary** (``transport="binary"``): the entry carries a ``handle`` and no
      ``data``; ``fetch_bytes(handle)`` must be supplied to pull the raw bytes
      from ``GET /v1/activations/{handle}``. :class:`VLLMLensClient` wires this
      automatically; a bare call without ``fetch_bytes`` raises a clear error
      rather than silently dropping the tensor.

    If no ``activations`` key is present, returns an empty dict.
    """
    raw = response_json.get("activations")
    if raw is None:
        return {}

    out: dict[str, Any] = {}
    for name, encoded in raw.items():
        if isinstance(encoded, dict) and "__json__" in encoded:
            out[name] = encoded["__json__"]
        elif "data" in encoded:
            out[name] = deserialize_tensor(encoded)
        elif "handle" in encoded:
            if fetch_bytes is None:
                raise ValueError(
                    f"activation {name!r} uses binary transport "
                    f"(handle={encoded['handle']!r}); pass fetch_bytes= or use "
                    "VLLMLensClient, which fetches it for you"
                )
            out[name] = tensor_from_bytes(fetch_bytes(encoded["handle"]), encoded)
        else:
            raise ValueError(
                f"activation {name!r} has neither 'data' nor 'handle' — "
                "unrecognised transport"
            )
    return out


_JSON_SAFE_TYPES = (str, int, float, bool, type(None))


def _serialize_value(v: Any) -> Any:
    """Serialize a single value from ``ctx.saved`` for JSON transport."""
    if isinstance(v, torch.Tensor):
        return {"__type__": "tensor", **serialize_tensor(v)}
    if isinstance(v, _JSON_SAFE_TYPES):
        return v
    if isinstance(v, (list, dict)):
        # Recurse for containers — fall through to cloudpickle if nested
        # values aren't JSON-safe.
        try:
            json.dumps(v)
            return v
        except (TypeError, ValueError):
            pass
    return {
        "__type__": "cloudpickle",
        "data": base64.b64encode(cloudpickle.dumps(v)).decode("ascii"),
    }


def _deserialize_value(v: Any) -> Any:
    """Deserialize a single value from serialized hook results."""
    if isinstance(v, dict) and "__type__" in v:
        if v["__type__"] == "tensor":
            return deserialize_tensor(v)
        if v["__type__"] == "cloudpickle":
            return cloudpickle.loads(base64.b64decode(v["data"]))
    return v


def serialize_hook_results(
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Serialize hook results for JSON transport.

    Input (from worker): ``{hook_index_str: ctx.saved_dict}``.
    Tensors are serialized via :func:`serialize_tensor`, JSON-safe values
    pass through, and other types are cloudpickle'd.
    """
    return {
        hook_idx: {k: _serialize_value(v) for k, v in saved.items()}
        for hook_idx, saved in results.items()
    }


def deserialize_hook_results(
    raw: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Deserialize hook results from JSON transport."""
    return {
        hook_idx: {k: _deserialize_value(v) for k, v in saved.items()}
        for hook_idx, saved in raw.items()
    }

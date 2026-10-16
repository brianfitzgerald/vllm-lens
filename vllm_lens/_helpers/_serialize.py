"""Tensor serialization helpers for vllm-lens activations and hook results."""

from __future__ import annotations

import base64
import ctypes
import json
import math
import mmap
import os
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

import cloudpickle
import numpy as np
import torch
import zstandard as zstd
from msgspec import msgpack

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


# vLLM's CUSTOM_TYPE_RAW_VIEW: its decoder returns this msgpack extension as a view of the
# received frame, with no copy and without VLLM_ALLOW_INSECURE_SERIALIZATION.
_RAW_VIEW = 3


class PackEntry(NamedTuple):
    """One tensor of a packed batch: its chunks, flattened in order, fill ``shape``."""

    req_id: str
    key: str
    dtype: torch.dtype
    shape: list[int]
    chunks: list[torch.Tensor]


# The index of (id, key, dtype, shape) and, per tensor, its memfd offset or raw-view extension,
# which vLLM decodes to a memoryview of the frame.
Packed = tuple[
    list[tuple[str, str, str, list[int]]], list[msgpack.Ext | memoryview | int]
]
# Every tensor in a shared segment starts on this byte boundary, so any dtype view is aligned.
_SHARED_ALIGN = 64
# MAP_POPULATE where the OS has it: under gVisor a first touch costs ~3 s per 1.2 GB, this ~0.3 s.
SHARED_MAP_FLAGS = mmap.MAP_SHARED | getattr(mmap, "MAP_POPULATE", 0)
# The memfd name; /proc shows it as the link target "/memfd:<name> (deleted)".
SHARED_NAME = "vllm-lens-capture"


def shared_size(entries: list[PackEntry]) -> int:
    """The bytes ``entries`` take in a shared segment, each tensor on a 64-byte boundary."""
    return sum(
        -(-math.prod(entry.shape) * entry.dtype.itemsize // _SHARED_ALIGN)
        * _SHARED_ALIGN
        for entry in entries
    )


def shared_segment(size: int) -> tuple[int, mmap.mmap] | None:
    """A memfd of ``size`` bytes and its shared mapping; None where the OS has no memfd.
    A memfd has no path, so its memory is freed when the last fd and mapping go."""
    fd = -1
    try:
        create = getattr(os, "memfd_create", None)
        if create is not None:
            fd = create(SHARED_NAME)
        else:
            # Some Python builds lack os.memfd_create; Linux libc has it. MFD_CLOEXEC = 1.
            fd = ctypes.CDLL(None).memfd_create(SHARED_NAME.encode(), 1)
        if fd < 0:
            return None
        os.ftruncate(fd, size)
        return fd, mmap.mmap(fd, size, flags=SHARED_MAP_FLAGS)
    except (AttributeError, OSError):
        if fd >= 0:
            os.close(fd)
        return None


def pack_chunks(
    entries: list[PackEntry], copy: bool = False, shared: mmap.mmap | None = None
) -> Packed:
    """Pack ``entries`` as an index of ``(id, key, dtype, shape)`` and, per tensor, its offset in
    ``shared`` or a raw-view msgpack extension; ``copy`` gives the extension bytes for a pickle."""
    index: list[tuple[str, str, str, list[int]]] = []
    buffers: list[msgpack.Ext | memoryview | int] = []
    offset = 0
    for entry in entries:
        chunks = [chunk.detach().reshape(-1) for chunk in entry.chunks]
        one = len(chunks) == 1 and chunks[0].stride(-1) == 1
        flat = chunks[0] if one else torch.cat(chunks)
        if flat.numel() != math.prod(entry.shape):
            raise ValueError(
                f"{entry.req_id} {entry.key}: chunks hold {flat.numel()} values, "
                f"shape {entry.shape} needs {math.prod(entry.shape)}"
            )
        data = flat.view(torch.uint8)
        index.append((entry.req_id, entry.key, str(entry.dtype), entry.shape))
        if shared is None:
            array = data.numpy()
            buffers.append(
                msgpack.Ext(_RAW_VIEW, array.tobytes() if copy else array.data)
            )
            continue
        if data.numel():
            torch.frombuffer(
                shared, dtype=torch.uint8, count=data.numel(), offset=offset
            ).copy_(data)
        buffers.append(offset)
        offset += -(-data.numel() // _SHARED_ALIGN) * _SHARED_ALIGN
    # A tuple: vLLM's typed utility results leave its items as they arrive.
    return index, buffers


def unpack_payloads(
    packed: Packed, shared: mmap.mmap | None = None
) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    """Rebuild ``{id: {"activations": {key: tensor}}}`` from :func:`pack_chunks`; each tensor is
    a view of only its own bytes in ``shared`` or the frame, so a pickle carries no others."""
    out: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for (req_id, key, dtype, shape), buffer in zip(*packed):
        torch_dtype = getattr(torch, dtype.removeprefix("torch."))
        if isinstance(buffer, int):
            assert shared is not None
            size = math.prod(shape) * torch_dtype.itemsize
            data = memoryview(shared)[buffer : buffer + size]
        else:
            data = buffer.data if isinstance(buffer, msgpack.Ext) else buffer
        if math.prod(shape) == 0:
            tensor = torch.empty(shape, dtype=torch_dtype)
        else:
            with warnings.catch_warnings():
                # A received frame is ours alone; torch warns only because it is read-only.
                warnings.simplefilter("ignore", UserWarning)
                tensor = torch.frombuffer(data, dtype=torch.uint8)
            tensor = tensor.view(torch_dtype).reshape(shape)
        out.setdefault(req_id, {"activations": {}})["activations"][key] = tensor
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


def decode_layers(value: Any) -> list[int] | None:
    """The layers of an ``output_residual_stream`` or ``output_qk`` value; None for every
    layer. ``vllm_xargs`` sends the list as a JSON string."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    return [int(layer) for layer in value] if isinstance(value, list) else None


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

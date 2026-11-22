"""Tensor serialization helpers for vllm-lens activations."""

from __future__ import annotations

import base64
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from vllm_lens._helpers._activation_store import ActivationStore


# torch dtypes that numpy cannot represent natively.
_TORCH_TO_NUMPY_VIEW: dict[torch.dtype, np.dtype[Any]] = {
    torch.bfloat16: np.dtype(np.int16),
}


def _encode_tensor(tensor: torch.Tensor) -> tuple[bytes, dict[str, Any]]:
    """A tensor's bytes and the metadata needed to round-trip it; activations barely
    compress, so they are not compressed.

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

    meta = {
        "dtype": str(arr.dtype),
        "original_dtype": original_dtype,
        "shape": list(arr.shape),
    }
    return arr.tobytes(), meta


def serialize_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    """A tensor as a JSON-serializable dict: base64 ``data`` of its native-dtype bytes,
    the numpy storage ``dtype``, ``original_dtype`` and ``shape``."""
    raw, meta = _encode_tensor(tensor)
    return {"data": base64.b64encode(raw).decode("ascii"), **meta}


def deserialize_tensor(d: dict[str, Any]) -> torch.Tensor:
    """The tensor of a :func:`serialize_tensor` dict."""
    return tensor_from_bytes(base64.b64decode(d["data"]), d)


def tensor_from_bytes(raw: bytes, meta: dict[str, Any]) -> torch.Tensor:
    """Reconstruct a tensor from its transport bytes and metadata, for the base64 and the
    binary-handle transports."""
    arr = np.frombuffer(raw, dtype=np.dtype(meta["dtype"])).reshape(meta["shape"])
    t = torch.from_numpy(arr.copy())
    original_dtype = meta.get("original_dtype")
    if original_dtype == "torch.bfloat16":
        t = t.view(torch.bfloat16)
    elif original_dtype == "torch.float16":
        t = t.to(torch.float16)
    return t


def serialize_activations(tensor_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert a flat activations dict to a JSON-serializable form.

    Tensors keep the existing wire format (see :func:`serialize_tensor`);
    JSON-safe non-tensor values (e.g. ``residual_stream_positions``) are wrapped
    as ``{"__json__": value}``.

    Input::

        {"residual_stream": Tensor(n_layers, total_pos, d), "residual_stream_positions": [0, 4]}

    Output::

        {"residual_stream": {"data": "<b64>", ...}, "residual_stream_positions": {"__json__": [0, 4]}}
    """
    return {
        name: serialize_tensor(t) if isinstance(t, torch.Tensor) else {"__json__": t}
        for name, t in tensor_dict.items()
    }


def serialize_tensor_binary(
    tensor: torch.Tensor, store: ActivationStore
) -> dict[str, Any]:
    """Park a tensor's bytes in ``store``; return a JSON handle descriptor.

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
    # Non-tensor entries ride in the JSON as usual.
    for name, t in tensor_dict.items():
        if not isinstance(t, torch.Tensor):
            out[name] = {"__json__": t}
    return out


def decode_activations(
    response_json: dict[str, Any],
    *,
    fetch_bytes: Callable[[str], bytes] | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Decode activations from an HTTP API response.

    Takes the raw JSON response dict from ``/v1/completions`` or
    ``/v1/chat/completions`` and converts any activation tensors back to
    ``torch.Tensor``; non-tensor ``{"__json__": ...}`` entries pass through as-is.
    Three transports are supported per tensor entry:

    - **base64** (default): the entry carries ``data``; decoded in-process.
    - **binary** (``transport="binary"``): the entry carries a ``handle`` and no
      ``data``; ``fetch_bytes(handle)`` must be supplied to pull the raw bytes
      from ``GET /v1/activations/{handle}``. :class:`VLLMLensClient` wires this
      automatically; a bare call without ``fetch_bytes`` raises a clear error
      rather than silently dropping the tensor.
    - **file** (``transport="file"``): the entry carries the ``path`` of the capture
      file the server wrote; this waits up to ``timeout`` seconds for it and maps it.
      The caller removes the file.

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
        elif encoded.get("transport") == "file":
            from vllm_lens._helpers._capture_file import read_capture, wait_capture

            wait_capture(encoded["path"], timeout)
            capture = read_capture(encoded["path"])
            rows = encoded["rows"]
            positions = capture.get("residual_stream_positions")
            if positions is not None:
                rows = sum(1 for pos in positions if pos < rows)
                out["residual_stream_positions"] = positions[:rows]
            out[name] = capture["residual_stream"][:, :rows]
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

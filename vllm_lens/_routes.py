"""FastAPI routes of vllm-lens: binary activation handles.

Separate module so that ``from __future__ import annotations`` in
``_activations_plugin.py`` does not interfere with FastAPI's
annotation-based dependency injection.
"""

import json
from collections.abc import Iterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

# Binary activation transport (issue #31). Separate prefix so it can be included
# independently of the hook-management router.
activations_router = APIRouter(prefix="/v1/activations", tags=["vllm-lens"])


_ACT_CHUNK_BYTES = 1 << 20  # 1 MiB


@activations_router.get("/{handle}")
async def get_activation(handle: str) -> StreamingResponse:
    """Stream a binary activation payload parked by an ``activations_transport="binary"`` capture.

    Returns the raw ``zstd`` tensor bytes as ``application/octet-stream``; the
    dtype/shape metadata needed to reconstruct the tensor rides in
    ``X-VLLM-Lens-*`` headers (and is also echoed in the completion JSON that
    handed out the handle, so a client already has it). An unknown or expired
    handle is a flat 404 — indistinguishable from a never-existed one, so it
    leaks nothing. ``handle`` is only ever a dict key here (no filesystem, no
    ``collective_rpc``), so there is no traversal or injection surface.

    Auth: this router is included into vLLM's OpenAI app, whose
    ``AuthenticationMiddleware`` guards every ``/v1`` path — so this endpoint
    requires the same API key as ``/v1/completions``. The handle is an
    unguessable (192-bit) bearer capability scoped to a single capture; with a
    single shared server API key it is the per-request boundary. ``no-store``
    keeps intermediaries from caching model internals; per-principal
    ownership-binding for multi-tenant keying is left as a follow-up (issue #31).

    The payload is streamed in chunks rather than copied whole into a second
    response buffer, so serving a multi-GB capture doesn't double peak memory.
    """
    from vllm_lens._helpers._activation_store import store

    result = store.get(handle)
    if result is None:
        raise HTTPException(404, "Unknown or expired activation handle")
    payload, meta = result
    headers = {
        "Content-Length": str(len(payload)),
        "Cache-Control": "no-store, private",
        "X-Content-Type-Options": "nosniff",
        "X-VLLM-Lens-Dtype": str(meta.get("dtype", "")),
        "X-VLLM-Lens-Original-Dtype": str(meta.get("original_dtype", "")),
        "X-VLLM-Lens-Shape": json.dumps(meta.get("shape", [])),
        "X-VLLM-Lens-Compression": str(meta.get("compression", "")),
    }

    def _chunks() -> "Iterator[bytes]":
        for i in range(0, len(payload), _ACT_CHUNK_BYTES):
            yield payload[i : i + _ACT_CHUNK_BYTES]

    return StreamingResponse(
        _chunks(), media_type="application/octet-stream", headers=headers
    )

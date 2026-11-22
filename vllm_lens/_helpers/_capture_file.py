"""One request's captured rows as a safetensors file. The worker writes each row at its file
row into ``<key>.partial``, then writes the header and renames the file to
``<key>.safetensors``; readers wait for that name."""

from __future__ import annotations

import asyncio
import hashlib
import json
import mmap
import os
import time
from typing import Any

import torch

CAPTURE_DIR_ENV = "VLLM_LENS_CAPTURE_DIR"
# The header area before the rows; the JSON is padded with spaces to fill it.
HEADER_BYTES = 4096
_DTYPES = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32"}
_TORCH_DTYPES = {name: dtype for dtype, name in _DTYPES.items()}


def capture_path(
    directory: str, key: str, request_id: str, *, partial: bool = False
) -> str:
    """The partial or sealed file of the request with external id ``request_id`` and capture
    key ``key``; a key can be shared by the requests of one offline call."""
    name = f"{key}-{hashlib.sha1(request_id.encode()).hexdigest()[:16]}"
    return os.path.join(directory, f"{name}.{'partial' if partial else 'safetensors'}")


def seal(
    fd: int,
    path: str,
    *,
    dtype: torch.dtype,
    rows: int,
    layers: list[int],
    hidden: int,
) -> None:
    """Write the header of ``rows`` rows of ``[layers, hidden]``, cut the file there, close
    it and rename it from its partial name."""
    data = rows * len(layers) * hidden * dtype.itemsize
    tensors: dict[str, Any] = {
        "__metadata__": {"layers": json.dumps(layers)},
        "residual_stream": {
            "dtype": _DTYPES[dtype],
            "shape": [rows, len(layers), hidden],
            "data_offsets": [0, data],
        },
    }
    header = json.dumps(tensors).encode()
    if len(header) > HEADER_BYTES - 8:
        raise ValueError("The capture header does not fit its area")
    header = header.ljust(HEADER_BYTES - 8, b" ")
    os.pwrite(fd, len(header).to_bytes(8, "little") + header, 0)
    os.ftruncate(fd, HEADER_BYTES + data)
    os.close(fd)
    os.rename(path, path.removesuffix(".partial") + ".safetensors")


def read_capture(path: str) -> dict[str, Any]:
    """Map a sealed capture as ``residual_stream`` ``[layers, rows, hidden]``, a view of the
    file."""
    with open(path, "rb") as file:
        size = os.fstat(file.fileno()).st_size
        mapped = mmap.mmap(
            file.fileno(),
            size,
            flags=mmap.MAP_PRIVATE | getattr(mmap, "MAP_POPULATE", 0),
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
    length = int.from_bytes(mapped[:8], "little")
    header = json.loads(mapped[8 : 8 + length])
    entry = header["residual_stream"]
    low, high = entry["data_offsets"]
    values = torch.frombuffer(
        mapped, dtype=torch.uint8, count=high - low, offset=HEADER_BYTES + low
    )
    rows = values.view(_TORCH_DTYPES[entry["dtype"]]).view(entry["shape"])
    return {"residual_stream": rows.permute(1, 0, 2)}


def wait_capture(path: str, timeout: float) -> None:
    """Wait until the worker seals the capture at ``path``."""
    deadline = time.monotonic() + timeout
    while not os.path.exists(path):
        if time.monotonic() > deadline:
            raise TimeoutError(f"The capture {path} was not written in {timeout} s")
        time.sleep(0.002)


async def wait_capture_async(path: str, timeout: float) -> None:
    """Wait until the worker seals the capture at ``path``, without blocking the loop."""
    deadline = time.monotonic() + timeout
    while not os.path.exists(path):
        if time.monotonic() > deadline:
            raise TimeoutError(f"The capture {path} was not written in {timeout} s")
        await asyncio.sleep(0.002)

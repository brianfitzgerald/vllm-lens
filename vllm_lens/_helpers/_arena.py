"""Captured rows in a shared-memory arena, after vLLM's IPC cache: the worker allocates one
chunk per request from vLLM's ``SingleWriterShmRingBuffer``, the GPU copies rows into it, and
at seal the worker writes the chunk header and a small marker file that names the chunk.
Readers map the arena by name and set the chunk's released byte when done."""

from __future__ import annotations

import asyncio
import atexit
import hashlib
import json
import os
import struct
import time
import uuid
from multiprocessing import shared_memory
from typing import Any
from unittest.mock import patch

import torch

CAPTURE_DIR_ENV = "VLLM_LENS_CAPTURE_DIR"
# Each chunk starts on this boundary, so its rows start one boundary later.
ALIGN = 4096
# vLLM's chunk metadata: a 4-byte monotonic id and a 4-byte signed size.
_CHUNK_MD = 8
# The header in a chunk's data: ready and released bytes, the JSON length, the seal time
# and the JSON; the rows start right after it, on an ALIGN boundary.
HEADER_BYTES = ALIGN - _CHUNK_MD
_HEADER = struct.Struct("<BBxxId")
# vLLM's chunk size field is a signed 32-bit int.
MAX_CHUNK_BYTES = (1 << 31) - ALIGN
_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
# The arenas this process has opened, by name.
_OPENED: dict[str, shared_memory.SharedMemory] = {}


def capture_path(
    directory: str, key: str, request_id: str, *, partial: bool = False
) -> str:
    """The marker of the request with external id ``request_id`` and capture key ``key``; a
    key can be shared by the requests of one offline call."""
    name = f"{key}-{hashlib.sha1(request_id.encode()).hexdigest()[:16]}"
    return os.path.join(directory, f"{name}.lens{'.partial' if partial else ''}")


class Arena:
    """The writer's arena: vLLM's single-writer shared-memory ring, registered with CUDA so
    the GPU copies rows straight into it."""

    def __init__(self, gib: float, timeout: float) -> None:
        from vllm.distributed.device_communicators.shm_object_storage import (
            SingleWriterShmRingBuffer,
        )

        size = int(gib * (1 << 30)) // ALIGN * ALIGN
        self.ring = SingleWriterShmRingBuffer(
            size, name=f"vllm-lens-{uuid.uuid4().hex[:16]}", create=True
        )
        self.name = self.ring.shared_memory.name
        self.size = size
        self.timeout = timeout
        self.bytes = torch.frombuffer(
            self.ring.shared_memory.buf, dtype=torch.uint8, count=size
        )
        code = torch.cuda.cudart().cudaHostRegister(self.bytes.data_ptr(), size, 0)
        if int(code) != 0:
            raise RuntimeError(f"cudaHostRegister of the capture arena failed: {code}")
        self._warned = False
        # Remove the shared memory when the process exits.
        atexit.register(self.ring.close)

    def _is_free(self, _id: int, data: memoryview) -> bool:
        """Whether a chunk's reader released it, or it was sealed longer ago than the
        timeout (its reader is gone)."""
        ready, released, _length, sealed = _HEADER.unpack_from(data)
        return bool(released) or (bool(ready) and time.time() - sealed > self.timeout)

    def allocate(self, nbytes: int) -> tuple[int, int, int]:
        """A chunk of at least ``nbytes`` data bytes, as its address, monotonic id and data
        offset in the arena; waits for readers to release the oldest chunks."""
        size = -(-(nbytes + _CHUNK_MD) // ALIGN) * ALIGN - _CHUNK_MD
        while True:
            try:
                address, ident = self.ring.allocate_buf(size)
                break
            except MemoryError:
                self.ring.free_buf(self._is_free, size + _CHUNK_MD)
            try:
                address, ident = self.ring.allocate_buf(size)
                break
            except MemoryError:
                if not self._warned:
                    self._warned = True
                    import logging

                    logging.getLogger("vllm.lens").warning(
                        "vllm-lens: the capture arena is full; waiting for readers to "
                        "release captures (capture_arena_gib sets its size)."
                    )
                time.sleep(0.001)
        offset = address % self.size + _CHUNK_MD
        # A reused chunk may hold an old header.
        self.bytes[offset : offset + _HEADER.size].zero_()
        return address, ident, offset

    def seal(
        self,
        marker: str,
        address: int,
        ident: int,
        offset: int,
        header: dict[str, Any],
        positions: list[int] | None,
        positions_offset: int,
    ) -> None:
        """Write the positions and the header of a chunk, set its ready byte last, then write
        its marker."""
        view = self.bytes.numpy()
        if positions:
            data = struct.pack(f"<{len(positions)}q", *positions)
            view[positions_offset : positions_offset + len(data)] = memoryview(data)
        body = json.dumps(header).encode()
        if len(body) > HEADER_BYTES - _HEADER.size:
            raise ValueError("The capture header does not fit its area")
        view[offset + _HEADER.size : offset + _HEADER.size + len(body)] = memoryview(
            body
        )
        _HEADER.pack_into(view, offset, 0, 0, len(body), time.time())
        view[offset] = 1
        write_marker(
            marker,
            {"name": self.name, "size": self.size, "address": address, "id": ident},
        )

    def close(self) -> None:
        """Unregister and remove the arena."""
        torch.cuda.cudart().cudaHostUnregister(self.bytes.data_ptr())
        self.ring.close()


def write_marker(path: str, marker: dict[str, Any]) -> None:
    """Write a capture's marker under its partial name and rename it to its final name."""
    partial = path + ".partial"
    with open(partial, "w") as file:
        json.dump(marker, file)
    os.rename(partial, path)


def _chunk(marker: dict[str, Any]) -> tuple[memoryview, int]:
    """The arena of a marker and the data offset of its chunk, checked against the chunk's
    monotonic id."""
    if "error" in marker:
        raise RuntimeError(marker["error"])
    name = marker["name"]
    arena = _OPENED.get(name)
    if arena is None:
        # Python tracks shared memory it did not create; vLLM's readers patch this out too.
        with patch("multiprocessing.resource_tracker.register", lambda *a, **k: None):
            arena = _OPENED[name] = shared_memory.SharedMemory(name=name)
    buf = arena.buf
    assert buf is not None
    start = marker["address"] % marker["size"]
    if int.from_bytes(buf[start : start + 4], "little", signed=True) != marker["id"]:
        raise RuntimeError("The capture was released or overwritten")
    return buf, start + _CHUNK_MD


def read_capture(path: str) -> dict[str, Any]:
    """Map the capture of a marker as ``residual_stream`` ``[layers, rows, hidden]`` (a view
    of the arena, valid until release) and, for a position selector,
    ``residual_stream_positions``."""
    with open(path) as file:
        marker = json.load(file)
    buf, offset = _chunk(marker)
    ready, _released, length, _sealed = _HEADER.unpack_from(buf, offset)
    if not ready:
        raise RuntimeError("The capture is not sealed")
    header = json.loads(
        bytes(buf[offset + _HEADER.size : offset + _HEADER.size + length])
    )
    dtype = _DTYPES[header["dtype"]]
    shape = (header["rows"], len(header["layers"]), header["hidden"])
    count = shape[0] * shape[1] * shape[2]
    rows = torch.frombuffer(
        buf, dtype=dtype, count=count, offset=offset + HEADER_BYTES
    ).view(shape)
    activations: dict[str, Any] = {"residual_stream": rows.permute(1, 0, 2)}
    if header.get("positions") is not None:
        low = header["positions_offset"]
        activations["residual_stream_positions"] = list(
            struct.unpack_from(f"<{header['positions']}q", buf, low)
        )
    return activations


def release_capture(path: str) -> None:
    """Set the released byte of a marker's chunk, so the worker can reuse it, and remove the
    marker."""
    try:
        with open(path) as file:
            marker = json.load(file)
        if "error" not in marker:
            try:
                buf, offset = _chunk(marker)
                buf[offset + 1] = 1
            except RuntimeError:
                pass
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def load_capture(path: str) -> dict[str, Any]:
    """Read a capture into process memory, then release it."""
    try:
        activations = read_capture(path)
        activations["residual_stream"] = activations["residual_stream"].clone()
        return activations
    finally:
        release_capture(path)


def wait_capture(path: str, timeout: float) -> None:
    """Wait until the worker writes the marker at ``path``."""
    deadline = time.monotonic() + timeout
    while not os.path.exists(path):
        if time.monotonic() > deadline:
            raise TimeoutError(f"The capture {path} was not written in {timeout} s")
        time.sleep(0.001)


class _Watcher:
    """One task per event loop that scans the marker directories each millisecond and wakes
    the requests whose markers appeared, so the poll cost does not grow with the requests."""

    def __init__(self) -> None:
        self.waiting: dict[str, dict[str, asyncio.Future[None]]] = {}
        self.task: asyncio.Task[None] | None = None

    async def wait(self, path: str, timeout: float) -> None:
        """Wait for the marker at ``path``."""
        directory, name = os.path.split(path)
        future = asyncio.get_running_loop().create_future()
        self.waiting.setdefault(directory, {})[name] = future
        if self.task is None or self.task.done():
            self.task = asyncio.get_running_loop().create_task(self.run())
        try:
            await asyncio.wait_for(future, timeout)
        except TimeoutError:
            raise TimeoutError(
                f"The capture {path} was not written in {timeout} s"
            ) from None
        finally:
            self.waiting.get(directory, {}).pop(name, None)

    async def run(self) -> None:
        """Scan while any request waits."""
        while any(self.waiting.values()):
            for directory, names in list(self.waiting.items()):
                if not names:
                    continue
                for entry in os.scandir(directory):
                    future = names.get(entry.name)
                    if future is not None and not future.done():
                        future.set_result(None)
            await asyncio.sleep(0.001)


# One watcher per event loop.
_WATCHERS: dict[int, _Watcher] = {}


async def wait_capture_async(path: str, timeout: float) -> None:
    """Wait until the worker writes the marker at ``path``, without blocking the loop."""
    if os.path.exists(path):
        return
    loop = id(asyncio.get_running_loop())
    watcher = _WATCHERS.get(loop)
    if watcher is None:
        watcher = _WATCHERS[loop] = _Watcher()
    await watcher.wait(path, timeout)

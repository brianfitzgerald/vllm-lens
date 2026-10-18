"""The offline batch collect: workers write captures into a memfd of the calling process."""

from __future__ import annotations

import ctypes
import math
import mmap
import os
import secrets
from typing import NamedTuple

import torch

# Every tensor in a segment starts on this byte boundary, so any dtype view is aligned.
_ALIGN = 64
# MAP_POPULATE where the OS has it: under gVisor a first touch costs ~3 s per 1.2 GB, this ~0.3 s.
MAP_FLAGS = mmap.MAP_SHARED | getattr(mmap, "MAP_POPULATE", 0)


class PackEntry(NamedTuple):
    """One tensor of a batch: its chunks, flattened in order, fill ``shape``."""

    req_id: str
    key: str
    dtype: torch.dtype
    shape: list[int]
    chunks: list[torch.Tensor]


class Segment(NamedTuple):
    """A memfd of the caller: its pid, fd, unique name and size, as a worker opens it."""

    pid: int
    fd: int
    name: str
    size: int


# The index of (id, key, dtype, shape) and each tensor's offset in the segment.
Packed = tuple[list[tuple[str, str, str, list[int]]], list[int]]


def _aligned(size: int) -> int:
    """``size`` rounded up to the segment alignment."""
    return -(-size // _ALIGN) * _ALIGN


def shared_size(entries: list[PackEntry]) -> int:
    """The bytes ``entries`` take in a segment."""
    return sum(
        _aligned(math.prod(entry.shape) * entry.dtype.itemsize) for entry in entries
    )


def shared_segment(size: int) -> tuple[Segment, mmap.mmap]:
    """A new memfd of ``size`` bytes and its mapping; the memory is freed when the fd
    and the last view of the mapping are gone."""
    name = f"vllm-lens-{os.getpid()}-{secrets.token_hex(8)}"
    create = getattr(os, "memfd_create", None)
    if create is not None:
        fd = create(name)
    else:
        # python-build-standalone lacks os.memfd_create; Linux libc has it. MFD_CLOEXEC = 1.
        fd = ctypes.CDLL(None, use_errno=True).memfd_create(name.encode(), 1)
        if fd < 0:
            raise OSError(ctypes.get_errno(), "memfd_create failed")
    try:
        os.ftruncate(fd, size)
        region = mmap.mmap(fd, size, flags=MAP_FLAGS)
    except BaseException:
        os.close(fd)
        raise
    return Segment(os.getpid(), fd, name, size), region


def open_segment(segment: Segment) -> mmap.mmap:
    """Map the caller's ``segment`` in this process; raise if it is not reachable."""
    path = f"/proc/{segment.pid}/fd/{segment.fd}"
    try:
        # On another host or in another pid namespace the path names something else.
        if os.readlink(path) != f"/memfd:{segment.name} (deleted)":
            raise OSError(f"{path} is not the memfd {segment.name}")
        handle = os.open(path, os.O_RDWR)
        try:
            if os.fstat(handle).st_size != segment.size:
                raise OSError(f"{path} does not have {segment.size} bytes")
            return mmap.mmap(handle, segment.size, flags=MAP_FLAGS)
        finally:
            os.close(handle)
    except OSError as error:
        raise RuntimeError(
            "The offline capture collect needs each worker on the host and in the pid "
            f"namespace of the LLM process: {error}"
        ) from error


def pack_chunks(entries: list[PackEntry], shared: mmap.mmap | None) -> Packed:
    """Write ``entries`` into ``shared`` and return their index and offsets; ``shared`` is
    None only when every entry is empty."""
    index: list[tuple[str, str, str, list[int]]] = []
    offsets: list[int] = []
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
        if data.numel():
            assert shared is not None
            torch.frombuffer(
                shared, dtype=torch.uint8, count=data.numel(), offset=offset
            ).copy_(data)
        index.append((entry.req_id, entry.key, str(entry.dtype), entry.shape))
        offsets.append(offset)
        offset += _aligned(data.numel())
    return index, offsets


def unpack_payloads(
    packed: Packed, shared: mmap.mmap | None
) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    """Rebuild ``{id: {"activations": {key: tensor}}}`` from :func:`pack_chunks`; each tensor
    is a view of its own bytes in ``shared``, which the views keep mapped."""
    out: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for (req_id, key, dtype, shape), offset in zip(*packed):
        torch_dtype = getattr(torch, dtype.removeprefix("torch."))
        count = math.prod(shape) * torch_dtype.itemsize
        if count:
            assert shared is not None
            tensor = torch.frombuffer(
                shared, dtype=torch.uint8, count=count, offset=offset
            )
            tensor = tensor.view(torch_dtype).reshape(shape)
        else:
            tensor = torch.empty(shape, dtype=torch_dtype)
        out.setdefault(req_id, {"activations": {}})["activations"][key] = tensor
    return out

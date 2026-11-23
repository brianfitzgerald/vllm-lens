"""Fixtures for the plugin tests without vLLM or CUDA."""

import collections
import sys
from contextlib import suppress
from multiprocessing import shared_memory
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

# Layers 0 to 3 armed for capture and steering.
ARMED = SimpleNamespace(
    additional_config={
        "vllm_lens": {"capture_layers": [0, 1, 2, 3], "steer_layers": [0, 1, 2, 3]}
    },
    model_config=SimpleNamespace(hf_text_config=SimpleNamespace(num_hidden_layers=4)),
)


def _engine(collective_rpc: Any, *, offline: bool, **attrs: Any) -> SimpleNamespace:
    """An offline ``LLM`` or an ``AsyncLLM`` stand-in on the armed config."""
    config = attrs.pop("vllm_config", ARMED)
    where = (
        {"llm_engine": SimpleNamespace(vllm_config=config)}
        if offline
        else {"vllm_config": config}
    )
    return SimpleNamespace(
        collective_rpc=collective_rpc,
        _lens_steering_tables={},
        _lens_table_users=collections.Counter(),
        **where,
        **attrs,
    )


@pytest.fixture
def stub_engine():
    """The factory of engine stand-ins."""
    return _engine


class _FakeRing:
    """vLLM's ``SingleWriterShmRingBuffer`` reduced to linear allocation, with the same
    ``[4-byte id][4-byte size]`` chunk metadata."""

    def __init__(self, size: int, name: str, create: bool) -> None:
        self.shared_memory = shared_memory.SharedMemory(
            create=create, size=size, name=name
        )
        self.size = size
        self.end = 0
        self.next_id = 0

    def allocate_buf(self, size: int) -> tuple[int, int]:
        size += 8
        if self.end + size > self.size:
            raise MemoryError("full")
        address, ident = self.end, self.next_id
        buf = self.shared_memory.buf
        assert buf is not None
        buf[address : address + 4] = ident.to_bytes(4, "little", signed=True)
        buf[address + 4 : address + 8] = size.to_bytes(4, "little", signed=True)
        self.end += size
        self.next_id += 1
        return address, ident

    def free_buf(self, is_free_fn: Any, nbytes: int | None = None) -> range:
        return range(0)

    def close(self) -> None:
        with suppress(FileNotFoundError):
            self.shared_memory.close()
            self.shared_memory.unlink()


@pytest.fixture
def arena(monkeypatch):
    """A 1 MiB capture arena on a fake vLLM ring buffer, with CUDA registration stubbed."""
    from vllm_lens._helpers._arena import Arena

    name = "vllm.distributed.device_communicators.shm_object_storage"
    for depth in range(1, name.count(".") + 1):
        parent = ".".join(name.split(".")[:depth])
        if parent not in sys.modules:
            monkeypatch.setitem(sys.modules, parent, ModuleType(parent))
    module = ModuleType(name)
    module.SingleWriterShmRingBuffer = _FakeRing  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, name, module)
    cudart = SimpleNamespace(
        cudaHostRegister=lambda *a: 0, cudaHostUnregister=lambda *a: 0
    )
    monkeypatch.setattr(torch.cuda, "cudart", lambda: cudart)
    created = Arena(1 / 1024, timeout=120.0)
    yield created
    created.ring.close()

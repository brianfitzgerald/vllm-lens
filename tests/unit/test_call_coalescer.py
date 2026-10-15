"""The online path's worker calls go out together: calls queued while one RPC is in flight
share the next, each caller gets its own per-rank results, and a failed call fails alone."""

import asyncio
from typing import Any

import pytest

from vllm_lens._activations_plugin import _CallCoalescer
from vllm_lens._helpers._serialize import RUN_CALLS_ERROR


class _Engine:
    """Two ranks that echo each call's arguments, after letting other tasks run."""

    def __init__(self) -> None:
        self.batches: list[list[tuple[str, tuple[Any, ...]]]] = []

    async def collective_rpc(
        self, method: str, args: tuple[Any, ...]
    ) -> list[list[Any]]:
        assert method == "run_calls"
        (calls,) = args
        self.batches.append(calls)
        await asyncio.sleep(0.01)
        return [
            [
                (RUN_CALLS_ERROR, f"{name} failed")
                if name == "fail"
                else (rank, name, a)
                for name, a in calls
            ]
            for rank in range(2)
        ]


async def test_concurrent_calls_share_one_rpc() -> None:
    """Ten calls queued together go out as one RPC, and each gets its own results."""
    engine = _Engine()
    coalescer = _CallCoalescer(engine)  # type: ignore[arg-type]
    results = await asyncio.gather(*(coalescer.call("get", (n,)) for n in range(10)))
    assert len(engine.batches) == 1
    for n, per_rank in enumerate(results):
        assert per_rank == [(0, "get", (n,)), (1, "get", (n,))]


async def test_calls_during_a_flight_go_in_the_next_rpc() -> None:
    """A call made while an RPC is in flight waits for it, then goes in the next one."""
    engine = _Engine()
    coalescer = _CallCoalescer(engine)  # type: ignore[arg-type]
    first = asyncio.ensure_future(coalescer.call("set", (1,)))
    await asyncio.sleep(0.001)
    later = await asyncio.gather(*(coalescer.call("get", (n,)) for n in range(3)))
    await first
    assert [len(batch) for batch in engine.batches] == [1, 3]
    assert later[2] == [(0, "get", (2,)), (1, "get", (2,))]


async def test_a_failed_call_raises_only_for_its_caller() -> None:
    """A call the worker reports as failed raises; the others in its RPC return."""
    coalescer = _CallCoalescer(_Engine())  # type: ignore[arg-type]
    ok, failed = await asyncio.gather(
        coalescer.call("get", (1,)), coalescer.call("fail", ()), return_exceptions=True
    )
    assert ok == [(0, "get", (1,)), (1, "get", (1,))]
    assert isinstance(failed, RuntimeError) and "fail failed" in str(failed)


async def test_an_rpc_error_reaches_every_caller_of_that_rpc() -> None:
    """When the RPC itself fails, every call in it raises that error."""

    class _Broken:
        async def collective_rpc(self, method: str, args: tuple[Any, ...]) -> list[Any]:
            raise ConnectionError("engine gone")

    coalescer = _CallCoalescer(_Broken())  # type: ignore[arg-type]
    outcomes = await asyncio.gather(
        *(coalescer.call("get", (n,)) for n in range(3)), return_exceptions=True
    )
    assert all(isinstance(o, ConnectionError) for o in outcomes)
    with pytest.raises(ConnectionError):
        await coalescer.call("get", (9,))

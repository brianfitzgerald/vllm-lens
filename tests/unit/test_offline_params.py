"""CPU tests for the plugin entry points: the caller's SamplingParams stay as passed."""

import asyncio
from types import SimpleNamespace

import pytest
import torch

import vllm_lens._activations_plugin as plugin
from vllm_lens import SteeringVector


class _FakeLLM:
    """Records the RPCs and raises on ``fail_on``."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.calls: list[str] = []
        self.fail_on = fail_on
        self._hooks_installed = True

    def collective_rpc(self, method: str, args: tuple = ()) -> list:
        """Record the method name."""
        self.calls.append(method)
        if method == self.fail_on:
            raise RuntimeError("rpc failed")
        return []


@pytest.mark.parametrize("fail", [None, "generate", "set_steering_data"])
def test_offline_generate_keeps_the_callers_params(
    monkeypatch: pytest.MonkeyPatch, fail: str | None
) -> None:
    """vLLM gets a copy with the vectors replaced by a sentinel; the caller's extra_args
    are not changed, and the worker state is cleared also when a step raises."""
    vector = SteeringVector(activations=torch.ones(1, 4), layer_indices=[0])
    extra = {"apply_steering_vectors": [vector]}
    params = SimpleNamespace(extra_args=dict(extra), skip_reading_prefix_cache=False)
    received: list = []

    def original(self, prompts, sampling_params, *args, **kwargs):
        received.append(sampling_params)
        if fail == "generate":
            raise RuntimeError("engine failed")
        return []

    monkeypatch.setattr(plugin, "_original_llm_generate", original)
    llm = _FakeLLM(fail_on=fail)
    if fail:
        with pytest.raises(RuntimeError, match="failed"):
            plugin._patched_llm_generate(llm, ["p"], params)  # type: ignore[arg-type]
    else:
        plugin._patched_llm_generate(llm, ["p"], params)  # type: ignore[arg-type]

    assert params.extra_args == extra and params.skip_reading_prefix_cache is False
    assert llm.calls == ["set_steering_data", "clear_request_state"]
    if fail != "set_steering_data":
        assert received[0] is not params
        assert received[0].extra_args == {"_steering_id": "_steer_0"}


def test_async_generate_drops_client_sentinels(monkeypatch: pytest.MonkeyPatch) -> None:
    """A client cannot send the ``_steering_id`` / ``_hook_id`` sentinels to the workers."""
    params = SimpleNamespace(
        extra_args={"_steering_id": "_steer_0", "_hook_id": "_hook_0", "keep": 1},
        skip_reading_prefix_cache=False,
    )
    received: list = []

    async def original(self, prompt, sampling_params, request_id, **kwargs):
        received.append(dict(sampling_params.extra_args))
        yield SimpleNamespace(finished=True)

    async def run() -> None:
        async for _ in plugin._patched_generate(object(), "p", params, "r"):  # type: ignore[arg-type]
            pass

    monkeypatch.setattr(plugin, "_original_generate", original)
    asyncio.run(run())
    assert received == [{"keep": 1}]

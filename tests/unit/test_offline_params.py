"""CPU tests for the plugin entry points: the caller's SamplingParams stay as passed."""

import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import vllm_lens._activations_plugin as plugin
from vllm_lens import Hook, SteeringVector


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


class _FakeAsyncEngine:
    """Record async worker RPCs."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._hooks_installed = True

    async def collective_rpc(self, method: str, args: tuple = ()) -> list:
        """Record the method name."""
        self.calls.append(method)
        return []


async def test_async_generate_drops_client_sentinels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A client cannot send the ``_steering_id`` / ``_hook_id`` sentinels to the workers."""
    params = SimpleNamespace(
        extra_args={"_steering_id": "_steer_0", "_hook_id": "_hook_0", "keep": 1},
        skip_reading_prefix_cache=False,
    )
    received: list = []

    async def original(self, prompt, sampling_params, request_id, **kwargs):
        received.append(dict(sampling_params.extra_args))
        yield SimpleNamespace(finished=True)

    monkeypatch.setattr(plugin, "_original_generate", original)
    engine = _FakeAsyncEngine()
    async for _ in plugin._patched_generate(engine, "p", params, "r"):  # type: ignore[arg-type]
        pass
    assert received == [{"keep": 1}]


@pytest.mark.parametrize("core_request", [False, True], ids=["prompt", "core-request"])
@pytest.mark.parametrize("concurrent", [False, True], ids=["sequential", "concurrent"])
async def test_async_generate_keeps_reused_params(
    monkeypatch, core_request, concurrent
):
    """Reused parameters keep their payloads, and each call gets independent copies."""
    module = ModuleType("vllm.v1.engine")
    module.EngineCoreRequest = SimpleNamespace
    monkeypatch.setitem(sys.modules, module.__name__, module)
    extra = {
        "output_residual_stream": [0],
        "apply_steering_vectors": [
            SteeringVector(activations=torch.ones(1, 4), layer_indices=[0])
        ],
        "apply_hooks": [Hook(fn=lambda ctx, h: None, layer_indices=[0])],
        "activations_transport": "binary",
    }
    params = SimpleNamespace(extra_args=dict(extra), skip_reading_prefix_cache=False)
    prompt = SimpleNamespace(sampling_params=params) if core_request else "p"
    received = []

    async def original(self, prompt, sampling_params, request_id, **kwargs):
        received.append((prompt, sampling_params))
        await asyncio.sleep(0)
        yield SimpleNamespace(finished=True)

    async def generate(request_id):
        supplied = SimpleNamespace(extra_args={}) if core_request else params
        async for _ in plugin._patched_generate(engine, prompt, supplied, request_id):
            pass

    monkeypatch.setattr(plugin, "_original_generate", original)
    engine = _FakeAsyncEngine()
    if concurrent:
        await asyncio.gather(generate("first"), generate("second"))
    else:
        await generate("first")
        await generate("second")

    assert params.extra_args == extra
    assert params.skip_reading_prefix_cache is False
    assert len(received) == 2
    copies = [sp for _, sp in received]
    assert copies[0] is not copies[1]
    assert copies[0].extra_args is not copies[1].extra_args
    for copied_prompt, sp in received:
        assert sp is not params and sp.skip_reading_prefix_cache is True
        assert set(sp.extra_args) == {"output_residual_stream"}
        if core_request:
            assert copied_prompt is not prompt
            assert copied_prompt.sampling_params is sp
            assert prompt.sampling_params is params
    assert copies[0].extra_args == copies[1].extra_args
    for method in (
        "set_steering_data",
        "set_hook_data",
        "clear_request_state",
    ):
        assert engine.calls.count(method) == 2

"""CPU tests for the plugin entry points: the caller's SamplingParams stay as passed."""

import asyncio
from types import SimpleNamespace

import pytest
import torch

import vllm_lens._activations_plugin as plugin
from vllm_lens import SteeringVector
from vllm_lens._helpers._serialize import serialize_tensor


class _FakeLLM:
    """Records the RPCs and raises on ``fail_on``."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.calls: list[str] = []
        self.fail_on = fail_on
        self._hooks_installed = True
        model_config = SimpleNamespace(get_hidden_size=lambda: 4)
        self.llm_engine = SimpleNamespace(
            vllm_config=SimpleNamespace(model_config=model_config)
        )

    def collective_rpc(self, method: str, args: tuple = ()) -> list:
        """Record the method name."""
        self.calls.append(method)
        if method == self.fail_on:
            raise RuntimeError("rpc failed")
        return []


@pytest.mark.parametrize("fail", [None, "generate", "set_steering_data_batch"])
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
    assert llm.calls == ["set_steering_data_batch", "clear_request_state"]
    if fail != "set_steering_data_batch":
        assert received[0] is not params
        assert received[0].extra_args == {"_steering_id": "_steer_0"}


class _FakeAsyncEngine:
    """Records the async RPCs; the model has hidden size 4."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._hooks_installed = True
        model_config = SimpleNamespace(get_hidden_size=lambda: 4)
        self.vllm_config = SimpleNamespace(model_config=model_config)

    async def collective_rpc(self, method: str, args: tuple = ()) -> list:
        """Record the method name."""
        self.calls.append(method)
        return []


@pytest.mark.parametrize(
    ("extra", "sent", "calls"),
    [
        (
            {"_steering_id": "_steer_0", "_hook_id": "_hook_0", "keep": 1},
            {"keep": 1},
            [],
        ),
        (
            {"output_residual_stream_project": serialize_tensor(torch.ones(1, 4))},
            None,
            [],
        ),
    ],
    ids=["client-sentinels", "projection-without-capture"],
)
def test_async_generate_sends_only_what_it_releases(
    monkeypatch: pytest.MonkeyPatch, extra: dict, sent: dict | None, calls: list[str]
) -> None:
    """Client-sent sentinels are dropped, and directions are not sent for a request that
    does not capture, since only a capture releases them."""
    params = SimpleNamespace(extra_args=extra, skip_reading_prefix_cache=False)
    received: list = []

    async def original(self, prompt, sampling_params, request_id, **kwargs):
        received.append(dict(sampling_params.extra_args))
        yield SimpleNamespace(finished=True)

    async def run(engine: _FakeAsyncEngine) -> None:
        async for _ in plugin._patched_generate(engine, "p", params, "r"):  # type: ignore[arg-type]
            pass

    monkeypatch.setattr(plugin, "_original_generate", original)
    engine = _FakeAsyncEngine()
    asyncio.run(run(engine))
    assert engine.calls == calls
    if sent is not None:
        assert received == [sent]

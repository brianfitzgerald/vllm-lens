"""CPU tests for the offline entry points: the caller's SamplingParams stay as passed."""

from types import SimpleNamespace

import pytest
import torch

import vllm_lens._activations_plugin as plugin
from vllm_lens import SteeringVector


class _FakeLLM:
    """Records the RPCs; ``generate`` returns no outputs or raises."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._hooks_installed = True
        model_config = SimpleNamespace(get_hidden_size=lambda: 4)
        self.llm_engine = SimpleNamespace(
            vllm_config=SimpleNamespace(model_config=model_config)
        )

    def collective_rpc(self, method: str, args: tuple = ()) -> list:
        """Record the method name."""
        self.calls.append(method)
        return []


@pytest.mark.parametrize("fail", [False, True], ids=["returns", "raises"])
def test_offline_generate_keeps_the_callers_params(
    monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    """vLLM gets a copy with the vectors replaced by a sentinel; the caller's extra_args
    are not changed, and the steering data is cleared also when generate raises."""
    vector = SteeringVector(activations=torch.ones(1, 4), layer_indices=[0])
    extra = {"apply_steering_vectors": [vector]}
    params = SimpleNamespace(extra_args=dict(extra), skip_reading_prefix_cache=False)
    received: list = []

    def original(self, prompts, sampling_params, *args, **kwargs):
        received.append(sampling_params)
        if fail:
            raise RuntimeError("engine failed")
        return []

    monkeypatch.setattr(plugin, "_original_llm_generate", original)
    llm = _FakeLLM()
    if fail:
        with pytest.raises(RuntimeError, match="engine failed"):
            plugin._patched_llm_generate(llm, ["p"], params)  # type: ignore[arg-type]
    else:
        plugin._patched_llm_generate(llm, ["p"], params)  # type: ignore[arg-type]

    assert params.extra_args == extra and params.skip_reading_prefix_cache is False
    assert received[0] is not params
    assert received[0].extra_args == {"_steering_id": "_steer_0"}
    assert llm.calls == ["set_steering_data_batch", "clear_steering_data_batch"]

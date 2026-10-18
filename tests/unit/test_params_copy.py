"""The entry points do not change the caller's SamplingParams. No vLLM."""

from types import SimpleNamespace

import pytest
import torch

import vllm_lens._activations_plugin as plugin
from vllm_lens import SteeringVector


def _params() -> SimpleNamespace:
    """A SamplingParams stand-in with one steering vector."""
    vector = SteeringVector(activations=torch.ones(1, 4), layer_indices=[0])
    return SimpleNamespace(
        extra_args={"apply_steering_vectors": [vector]}, skip_reading_prefix_cache=False
    )


@pytest.mark.parametrize("many", [False, True], ids=["one", "list"])
def test_offline_generate_keeps_the_callers_params(
    monkeypatch: pytest.MonkeyPatch, many: bool
) -> None:
    """The copy carries the sentinels; the caller's params keep their vectors."""
    received: list = []
    monkeypatch.setattr(
        plugin,
        "_original_llm_generate",
        lambda self, prompts, params, *a, **k: received.append(params) or [],
    )
    llm = SimpleNamespace(
        _hooks_installed=True,
        llm_engine=SimpleNamespace(vllm_config=SimpleNamespace(additional_config=None)),
        collective_rpc=lambda method, args=(): [],
    )
    params = _params()
    plugin._patched_llm_generate(llm, ["p"], [params] if many else params)  # type: ignore[arg-type]
    assert "apply_steering_vectors" in params.extra_args
    assert "_steering_id" not in params.extra_args
    (sent,) = received
    sent = sent[0] if many else sent
    assert sent is not params and sent.extra_args["_steering_id"] == "_steer_0"
    assert "apply_steering_vectors" not in sent.extra_args


async def test_async_generate_keeps_the_callers_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The async entry point sends a copy and leaves the caller's params alone."""
    received: list = []

    async def original(self, prompt, params, request_id, **kwargs):
        received.append(params)
        yield SimpleNamespace(finished=True)

    monkeypatch.setattr(plugin, "_original_generate", original)

    async def collective_rpc(method: str, args: tuple = ()) -> list:
        return []

    engine = SimpleNamespace(
        vllm_config=SimpleNamespace(additional_config=None),
        _hooks_installed=True,
        collective_rpc=collective_rpc,
    )
    params = _params()
    async for _ in plugin._patched_generate(engine, "p", params, "r"):  # type: ignore[arg-type]
        pass
    assert "apply_steering_vectors" in params.extra_args
    (sent,) = received
    assert sent is not params and "apply_steering_vectors" not in sent.extra_args

"""Small-model HTTP checks of values, interventions, transports and cleanup.

Run with VLLM_TEST_MODEL=Qwen/Qwen2.5-0.5B-Instruct. Unlike the original
Llama-specific server tests, these infer hidden size from captured states.
"""

import uuid

import pytest
import torch

from vllm_lens import Hook, SteeringVector
from vllm_lens.client import VLLMLensClient

LAYER = 2
PROMPT = "The capital of France is"


@pytest.fixture
def client(vllm_server):
    client = VLLMLensClient(vllm_server)
    client.clear_hooks()
    yield client
    client.clear_hooks()


@pytest.mark.parametrize("endpoint", ["generate", "chat"])
@pytest.mark.parametrize("transport", ["base64", "binary"])
def test_capture_and_interventions(client, endpoint, transport):
    def generate(**kwargs):
        prompt = (
            PROMPT if endpoint == "generate" else [{"role": "user", "content": PROMPT}]
        )
        return getattr(client, endpoint)(
            prompt,
            max_tokens=1,
            temperature=0.0,
            capture_layers=[LAYER],
            activations_transport=transport,
            **kwargs,
        )

    baseline = generate().activations["residual_stream"]
    assert baseline.shape[0] == 1 and baseline.shape[1] > 1
    assert torch.isfinite(baseline).all() and baseline.abs().max() > 0.01

    def capture(ctx, h):
        ctx.saved.setdefault("parts", []).append(h.detach().cpu())

    captured = generate(hooks=[Hook(fn=capture, layer_indices=[LAYER])])
    hook_values = torch.cat(captured.hook_results["0"]["parts"], dim=0)
    native = captured.activations["residual_stream"][0]
    # vLLM may execute one surplus forward after stopping. Native capture is
    # trimmed; generic hook results intentionally retain every invocation.
    assert hook_values.shape[0] >= native.shape[0]
    torch.testing.assert_close(hook_values[: native.shape[0]], native, rtol=0, atol=0)

    vector = SteeringVector(
        activations=torch.ones(1, baseline.shape[-1]),
        layer_indices=[LAYER],
        scale=0.5,
    )
    steered = generate(steering_vectors=[vector]).activations["residual_stream"]
    # Same prompt, same capture/injection layer: directly check sign and scale.
    # Allow bf16 rounding, but a silent no-op (delta=0) must fail.
    error = (steered.float() - baseline.float() - 0.5).abs().mean()
    assert error < 0.02, f"incorrect steering delta: mean error {error}"

    def zero(ctx, h):
        return torch.zeros_like(h)

    modified = generate(hooks=[Hook(fn=zero, layer_indices=[LAYER])])
    assert modified.activations["residual_stream"].abs().max() < 0.02

    # Repeated identical prompts also exercise prefix-cache bypass. A previous
    # intervention must not contaminate the next request or its cached prefix.
    restored = generate()
    torch.testing.assert_close(
        restored.activations["residual_stream"], baseline, rtol=0.02, atol=0.02
    )
    assert not restored.hook_results


def test_persistent_hook_cleanup(client):
    def capture(ctx, h):
        ctx.saved["seen"] = True

    client.register_hooks([Hook(fn=capture, layer_indices=[LAYER])])
    client.generate(PROMPT, max_tokens=1)
    results = client.collect_hook_results()
    assert results and any(hooks["0"]["seen"] for hooks in results.values())
    client.clear_hook_results()
    assert not client.collect_hook_results()
    client.clear_hooks()
    client.generate(PROMPT, max_tokens=1)
    assert not client.collect_hook_results()


@pytest.mark.parametrize("endpoint", ["generate", "chat"])
@pytest.mark.parametrize("kind", ["steering", "hooks", "persistent"])
def test_intervention_cache_isolation(client, endpoint, kind):
    """HTTP interventions isolate writes without requesting activation capture."""
    prompt = uuid.uuid4().hex + " The quick brown fox jumps over the lazy dog." * 20
    inputs = prompt if endpoint == "generate" else [{"role": "user", "content": prompt}]
    run = getattr(client, endpoint)
    options = {
        "cache_salt": uuid.uuid4().hex,
        "return_token_ids": True,
        "max_tokens": 20,
        "temperature": 0.0,
    }
    hook = Hook(fn=lambda ctx, h: torch.zeros_like(h), layer_indices=[LAYER])
    intervention = {}
    if kind == "steering":
        probe = run(inputs, capture_layers=[LAYER], max_tokens=1)
        hidden_dim = probe.activations["residual_stream"].shape[-1]
        intervention["steering_vectors"] = [
            SteeringVector(
                activations=torch.ones(1, hidden_dim), layer_indices=[LAYER], scale=10.0
            )
        ]
    elif kind == "hooks":
        intervention["hooks"] = [hook]
    else:
        client.register_hooks([hook])
    try:
        for _ in range(2):
            result = run(inputs, **options, **intervention)
            assert result.raw["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    finally:
        if kind == "persistent":
            client.clear_hooks()
    first = run(inputs, **options)
    assert first.raw["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    repeat = run(inputs, **options)
    assert repeat.raw["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
    fresh = run(inputs, **{**options, "cache_salt": uuid.uuid4().hex})
    # A cache hit is not bitwise equal to a prefill, so only the two prefills must match.
    tokens = first.raw["choices"][0]["token_ids"]
    assert tokens is not None
    assert tokens == fresh.raw["choices"][0]["token_ids"]

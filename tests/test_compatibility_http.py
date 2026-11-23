"""Small-model HTTP checks of values, interventions, transports and cleanup.

Run with VLLM_TEST_MODEL=Qwen/Qwen2.5-0.5B-Instruct. Unlike the original
Llama-specific server tests, these infer hidden size from captured states.
"""

import uuid

import pytest
import torch

from vllm_lens import SteeringVector
from vllm_lens.client import VLLMLensClient

LAYER = 2
PROMPT = "The capital of France is"


@pytest.fixture
def client(vllm_server):
    return VLLMLensClient(vllm_server)


@pytest.mark.parametrize("endpoint", ["generate", "chat"])
@pytest.mark.parametrize("transport", ["base64", "binary", "shm"])
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

    # Repeated identical prompts also exercise prefix-cache bypass. A previous
    # intervention must not contaminate the next request or its cached prefix.
    restored = generate()
    torch.testing.assert_close(
        restored.activations["residual_stream"], baseline, rtol=0.02, atol=0.02
    )


@pytest.mark.parametrize("endpoint", ["generate", "chat"])
def test_intervention_cache_isolation(client, endpoint):
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
    probe = run(inputs, capture_layers=[LAYER], max_tokens=1)
    hidden_dim = probe.activations["residual_stream"].shape[-1]
    vectors = [
        SteeringVector(
            activations=torch.ones(1, hidden_dim), layer_indices=[LAYER], scale=10.0
        )
    ]
    for _ in range(2):
        result = run(inputs, **options, steering_vectors=vectors)
        assert result.raw["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    first = run(inputs, **options)
    assert first.raw["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    repeat = run(inputs, **options)
    assert repeat.raw["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
    fresh = run(inputs, **{**options, "cache_salt": uuid.uuid4().hex})
    # A cache hit is not bitwise equal to a prefill, so only the two prefills must match.
    tokens = first.raw["choices"][0]["token_ids"]
    assert tokens is not None
    assert tokens == fresh.raw["choices"][0]["token_ids"]

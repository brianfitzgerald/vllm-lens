"""Tests for VLLMLensClient."""

import torch

from vllm_lens import SteeringVector
from vllm_lens.client import VLLMLensClient


def _make_client(vllm_server: str) -> VLLMLensClient:
    return VLLMLensClient(vllm_server)


# ---------------------------------------------------------------------------
# Basic generate
# ---------------------------------------------------------------------------


def test_generate_returns_text(vllm_server):
    client = _make_client(vllm_server)
    output = client.generate("Hello", max_tokens=5)
    assert isinstance(output.text, str)
    assert len(output.text) > 0


def test_generate_with_logprobs(vllm_server):
    client = _make_client(vllm_server)
    output = client.generate("Hello", max_tokens=1, logprobs=5)
    assert output.logprobs is not None
    assert "top_logprobs" in output.logprobs


def test_model_auto_detected(vllm_server):
    client = VLLMLensClient(vllm_server)
    assert "Llama" in client.model or "llama" in client.model


def test_chat_returns_text(vllm_server):
    client = _make_client(vllm_server)
    output = client.chat(
        [{"role": "user", "content": "Say hello in one word."}],
        max_tokens=5,
    )
    assert isinstance(output.text, str)
    assert len(output.text) > 0


# ---------------------------------------------------------------------------
# Activation capture
# ---------------------------------------------------------------------------


def test_capture_layers(vllm_server):
    client = _make_client(vllm_server)
    output = client.generate("Hello", max_tokens=1, capture_layers=[15])
    assert output.activations is not None
    rs = output.activations["residual_stream"]
    assert rs.shape[0] == 1  # single layer
    assert rs.shape[2] == 4096


# ---------------------------------------------------------------------------
# Steering vectors
# ---------------------------------------------------------------------------


def test_steering_via_client(vllm_server):
    client = _make_client(vllm_server)

    baseline = client.generate("I think the best dessert is", max_tokens=20)

    sv = SteeringVector(
        activations=torch.randn(1, 4096),
        layer_indices=[15],
        scale=15.0,
        norm_match=True,
    )
    steered = client.generate(
        "I think the best dessert is", max_tokens=20, steering_vectors=[sv]
    )
    assert steered.text != baseline.text

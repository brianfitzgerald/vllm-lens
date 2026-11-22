"""Integration tests for vllm-lens features against a running vLLM server.

These test activation extraction, steering vectors and steering tables
system over the OpenAI-compatible HTTP API.
"""

import json

import requests
import torch

from vllm_lens import (
    SteeringVector,
    deserialize_tensor,
    serialize_tensor,
)

from .conftest import MODEL

# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------


def test_activation_extraction(vllm_server):
    resp = requests.post(
        f"{vllm_server}/v1/completions",
        json={
            "model": MODEL,
            "prompt": "The future of AI is",
            "max_tokens": 10,
            "temperature": 0.0,
            "vllm_xargs": {"output_residual_stream": "[15]"},
        },
    ).json()

    assert "error" not in resp, resp
    rs = deserialize_tensor(resp["activations"]["residual_stream"])
    assert rs.ndim == 3
    assert rs.shape[2] == 4096  # Llama 8B hidden dim


# ---------------------------------------------------------------------------
# Steering vectors
# ---------------------------------------------------------------------------


def test_steering_changes_output(vllm_server):
    prompt = "I think the best dessert is"

    baseline = requests.post(
        f"{vllm_server}/v1/completions",
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": 20,
            "temperature": 0.0,
        },
    ).json()

    sv = SteeringVector(
        activations=torch.randn(1, 4096),
        layer_indices=[15],
        scale=15.0,
        norm_match=True,
    )
    steered = requests.post(
        f"{vllm_server}/v1/completions",
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": 20,
            "temperature": 0.0,
            "vllm_xargs": {
                "apply_steering_vectors": json.dumps([sv.model_dump()]),
            },
        },
    ).json()

    assert steered["choices"][0]["text"] != baseline["choices"][0]["text"]


def test_steering_scale_zero_matches_baseline(vllm_server):
    prompt = "The capital of France is"

    baseline = requests.post(
        f"{vllm_server}/v1/completions",
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": 20,
            "temperature": 0.0,
        },
    ).json()

    sv = SteeringVector(
        activations=torch.randn(1, 4096),
        layer_indices=[15],
        scale=0.0,
    )
    steered = requests.post(
        f"{vllm_server}/v1/completions",
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": 20,
            "temperature": 0.0,
            "vllm_xargs": {
                "apply_steering_vectors": json.dumps([sv.model_dump()]),
            },
        },
    ).json()

    assert steered["choices"][0]["text"] == baseline["choices"][0]["text"]


def test_layer_filtering_respects_list(vllm_server):
    """output_residual_stream=[15] should capture only 1 layer, not all 32.

    Regression: string '[15]' from vllm_xargs was treated as truthy
    instead of parsed as a JSON list.
    """
    resp = requests.post(
        f"{vllm_server}/v1/completions",
        json={
            "model": MODEL,
            "prompt": "Hello",
            "max_tokens": 1,
            "temperature": 0.0,
            "vllm_xargs": {"output_residual_stream": "[15]"},
        },
    ).json()

    assert "error" not in resp, resp
    rs = deserialize_tensor(resp["activations"]["residual_stream"])
    assert rs.shape[0] == 1, (
        f"Expected 1 layer (layer 15 only), got {rs.shape[0]} layers"
    )


# ---------------------------------------------------------------------------
# Steering tables and parallel samples
# ---------------------------------------------------------------------------


def _complete(vllm_server, xargs: dict, **body):
    """One completion of a fixed prompt with the given ``vllm_xargs``."""
    return requests.post(
        f"{vllm_server}/v1/completions",
        json={
            "model": MODEL,
            "prompt": "My favourite animal is",
            "max_tokens": 16,
            **body,
            "vllm_xargs": xargs,
        },
    ).json()


def test_steering_rows_match_steering_vectors(vllm_server):
    """A row of a registered table steers as the same vector sent in the request."""
    vectors = torch.randn(2, 4096, generator=torch.Generator().manual_seed(0))
    registered = requests.post(
        f"{vllm_server}/v1/steering_tables",
        json={"name": "server-rows", "vectors": serialize_tensor(vectors)},
    )
    assert registered.status_code == 200, registered.text
    assert registered.json()["rows"] == 2
    try:
        row = SteeringVector(table="server-rows", row=1, layer_indices=[15], scale=6.0)
        vector = SteeringVector(activations=vectors[1:2], layer_indices=[15], scale=6.0)
        by_row = _complete(
            vllm_server,
            {"apply_steering_vectors": json.dumps([row.model_dump()])},
            temperature=0.0,
        )
        by_vector = _complete(
            vllm_server,
            {"apply_steering_vectors": json.dumps([vector.model_dump()])},
            temperature=0.0,
        )
        baseline = _complete(vllm_server, {}, temperature=0.0)
        assert by_row["choices"][0]["text"] == by_vector["choices"][0]["text"]
        assert by_row["choices"][0]["text"] != baseline["choices"][0]["text"]
    finally:
        requests.delete(f"{vllm_server}/v1/steering_tables/server-rows")
    missing = _complete(
        vllm_server, {"apply_steering_vectors": json.dumps([row.model_dump()])}
    )
    assert "not registered" in json.dumps(missing), missing


def test_every_parallel_sample_is_steered(vllm_server):
    """Each of n choices equals a single request with its seed and the same steering."""
    vector = SteeringVector(
        activations=torch.randn(1, 4096, generator=torch.Generator().manual_seed(1)),
        layer_indices=[15],
        scale=8.0,
    )
    xargs = {"apply_steering_vectors": json.dumps([vector.model_dump()])}
    parallel = _complete(vllm_server, xargs, n=3, seed=100, temperature=1.0)
    texts = [
        choice["text"]
        for choice in sorted(parallel["choices"], key=lambda c: c["index"])
    ]
    single = [
        _complete(vllm_server, xargs, seed=100 + index, temperature=1.0)["choices"][0][
            "text"
        ]
        for index in range(3)
    ]
    assert texts == single
    unsteered = _complete(vllm_server, {}, n=3, seed=100, temperature=1.0)
    assert [
        c["text"] for c in sorted(unsteered["choices"], key=lambda c: c["index"])
    ] != texts


def test_capture_with_parallel_samples_is_rejected(vllm_server):
    """Capture with n > 1 fails with a clear error."""
    response = _complete(
        vllm_server, {"output_residual_stream": "[15]"}, n=2, temperature=1.0
    )
    assert "n > 1" in json.dumps(response), response

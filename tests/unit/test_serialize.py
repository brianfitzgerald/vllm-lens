"""Round-trip tests for tensor serialization helpers."""

from __future__ import annotations

import base64

import pytest
import torch

from vllm_lens._helpers._serialize import (
    decode_activations,
    deserialize_tensor,
    serialize_activations,
    serialize_tensor,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_round_trip(dtype: torch.dtype) -> None:
    """serialize_tensor → deserialize_tensor preserves values for all dtypes."""
    original = torch.randn(4, 8, dtype=dtype)
    serialized = serialize_tensor(original)
    recovered = deserialize_tensor(serialized)

    assert recovered.dtype == dtype
    assert recovered.shape == original.shape
    assert torch.equal(recovered, original)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_round_trip_activations(dtype: torch.dtype) -> None:
    """serialize_activations → decode_activations preserves values."""
    tensors = {"residual_stream": torch.randn(2, 4, 8, dtype=dtype)}
    serialized = serialize_activations(tensors)
    response_json = {"activations": serialized}
    recovered = decode_activations(response_json)

    assert "residual_stream" in recovered
    assert recovered["residual_stream"].dtype == dtype
    assert torch.equal(recovered["residual_stream"], tensors["residual_stream"])


def test_tensors_are_not_compressed() -> None:
    """Serialized output holds the raw bytes and the original dtype."""
    t = torch.randn(4, 8)
    serialized = serialize_tensor(t)
    assert "compression" not in serialized and "original_dtype" in serialized
    assert len(base64.b64decode(serialized["data"])) == t.nelement() * t.element_size()


def test_backward_compat_legacy_format() -> None:
    """deserialize_tensor handles old format (no compression, no original_dtype)."""
    original = torch.randn(4, 8)
    arr = original.numpy()
    legacy = {
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
    }
    recovered = deserialize_tensor(legacy)
    assert recovered.dtype == torch.float32
    assert torch.equal(recovered, original)


def test_decode_activations_empty() -> None:
    """decode_activations returns empty dict when no activations key."""
    assert decode_activations({}) == {}
    assert decode_activations({"other": "data"}) == {}


def test_bfloat16_cpu_operations() -> None:
    """bfloat16 tensors returned by deserialization support CPU operations."""
    original = torch.randn(4, 8, dtype=torch.bfloat16)
    recovered = deserialize_tensor(serialize_tensor(original))

    result = recovered.float()
    assert result.dtype == torch.float32
    assert torch.allclose(result, original.float())

    dot = (recovered @ recovered.T).float()
    expected = (original @ original.T).float()
    assert torch.allclose(dot, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_original_dtype_field(dtype: torch.dtype) -> None:
    """original_dtype field is always set and round-trips correctly."""
    serialized = serialize_tensor(torch.randn(2, 3, dtype=dtype))
    assert serialized["original_dtype"] == str(dtype)


def test_round_trip_qk_activations() -> None:
    """A Q/K payload round-trips: 4-D bf16 tensors + __json__ metadata."""
    import json

    activations = {
        "attn_q": torch.randn(2, 5, 4, 8, dtype=torch.bfloat16),
        "attn_k": torch.randn(2, 5, 2, 8, dtype=torch.bfloat16),
        "qk_layers": [2, 7],
        "qk_meta": [
            {"scale": 0.125, "sliding_window": [-1, -1], "alibi_slopes": None},
            {"scale": 0.125, "sliding_window": [255, 0], "alibi_slopes": None},
        ],
    }
    serialized = serialize_activations(activations)
    # The wire format must be JSON-safe end to end (HTTP path).
    recovered = decode_activations({"activations": json.loads(json.dumps(serialized))})

    assert torch.equal(recovered["attn_q"], activations["attn_q"])
    assert torch.equal(recovered["attn_k"], activations["attn_k"])
    assert recovered["qk_layers"] == [2, 7]
    assert recovered["qk_meta"] == activations["qk_meta"]

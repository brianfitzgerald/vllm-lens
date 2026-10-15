"""Round-trip tests for tensor serialization helpers."""

from __future__ import annotations

import base64
import pickle

import pytest
import torch

from vllm_lens._helpers._serialize import (
    PackEntry,
    decode_activations,
    deserialize_tensor,
    pack_chunks,
    serialize_activations,
    serialize_tensor,
    unpack_payloads,
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


def test_zstd_compression_present() -> None:
    """Serialized output includes compression metadata."""
    t = torch.randn(4, 8)
    serialized = serialize_tensor(t)
    assert serialized["compression"] == "zstd"
    assert "original_dtype" in serialized


def test_zstd_reduces_size() -> None:
    """Compressed payload is smaller than uncompressed for typical tensors."""
    t = torch.zeros(32, 128, 4096)  # highly compressible
    serialized = serialize_tensor(t)
    compressed_size = len(base64.b64decode(serialized["data"]))
    raw_size = t.nelement() * t.element_size()
    assert compressed_size < raw_size / 2


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_packed_chunks_round_trip(dtype: torch.dtype) -> None:
    """Chunks fill their tensor in flat order, each tensor comes back in its own storage, and
    a float32 tensor after an odd-sized one unpacks because offsets are aligned."""
    layers = [
        [torch.randn(2, 3).to(dtype), torch.randn(1, 3).to(dtype)] for _ in range(2)
    ]
    one = torch.randn(1, 3)[:, 2]
    norms = torch.randn(1, 3)
    packed = pack_chunks(
        [
            PackEntry("a", "residual_stream", dtype, [2, 3, 3], layers[0] + layers[1]),
            PackEntry("a", "residual_stream_norm", torch.float32, [1, 3], [norms]),
            PackEntry("b", "residual_stream_norm", torch.float32, [1, 1], [one]),
        ]
    )
    recovered = unpack_payloads(pickle.loads(pickle.dumps(packed)))
    expected = torch.stack([torch.cat(layer) for layer in layers])
    stream = recovered["a"]["activations"]["residual_stream"]
    assert stream.dtype == dtype and torch.equal(stream, expected)
    assert stream.untyped_storage().nbytes() == stream.numel() * stream.element_size()
    assert torch.equal(recovered["a"]["activations"]["residual_stream_norm"], norms)
    assert torch.equal(recovered["b"]["activations"]["residual_stream_norm"], one[None])
    assert all(start % 64 == 0 for *_, start, _ in packed["packed_index"])


@pytest.mark.parametrize("rows", [1, 3], ids=["short", "long"])
def test_packed_chunks_refuse_a_wrong_row_count(rows: int) -> None:
    """Chunks that do not fill the shape exactly raise, in place of filling another entry."""
    entry = PackEntry(
        "a", "residual_stream", torch.float32, [2, 4], [torch.ones(rows, 4)]
    )
    with pytest.raises((ValueError, RuntimeError)):
        pack_chunks([entry])


def test_packed_chunks_empty() -> None:
    """An empty batch packs and unpacks to nothing."""
    assert unpack_payloads(pack_chunks([])) == {}

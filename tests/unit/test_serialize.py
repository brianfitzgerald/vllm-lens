"""Round-trip tests for tensor serialization helpers."""

from __future__ import annotations

import base64
import pickle
from pathlib import Path

import pytest
import torch

from vllm_lens._helpers._serialize import (
    decode_activations,
    deserialize_tensor,
    pack_chunks,
    pack_payloads,
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
def test_packed_payloads_round_trip(dtype: torch.dtype) -> None:
    """pack_payloads → pickle → unpack_payloads returns every tensor, each in its own storage."""
    payloads = {
        "0": {"activations": {"residual_stream": torch.randn(2, 5, 8).to(dtype)}},
        "1": {"activations": {"residual_stream": torch.randn(2, 1, 8).to(dtype)}},
        "2": {
            "activations": {
                "residual_stream_projection": torch.randn(1, 3, 4),
                "residual_stream_norm": torch.randn(1, 3),
            }
        },
    }
    recovered = unpack_payloads(pickle.loads(pickle.dumps(pack_payloads(payloads))))

    assert set(recovered) == set(payloads)
    for req_id, payload in payloads.items():
        for key, tensor in payload["activations"].items():
            got = recovered[req_id]["activations"][key]
            assert got.dtype == tensor.dtype and torch.equal(got, tensor)
            assert got.untyped_storage().nbytes() == got.numel() * got.element_size()


def test_packed_payloads_empty() -> None:
    """An empty batch packs and unpacks to nothing."""
    assert unpack_payloads(pack_payloads({})) == {}


def test_packed_chunks_fill_each_tensor_in_order() -> None:
    """Chunks fill their tensor in flat order; a float32 tensor after an odd-sized bf16 one
    still unpacks, because every tensor starts on an aligned offset."""
    layer0 = [
        torch.randn(2, 3).to(torch.bfloat16),
        torch.randn(1, 3).to(torch.bfloat16),
    ]
    layer1 = [
        torch.randn(2, 3).to(torch.bfloat16),
        torch.randn(1, 3).to(torch.bfloat16),
    ]
    norms = torch.randn(1, 3)
    packed = pack_chunks(
        [
            ("a", "residual_stream", torch.bfloat16, [2, 3, 3], layer0 + layer1),
            ("b", "residual_stream_norm", torch.float32, [1, 3], [norms]),
        ]
    )
    recovered = unpack_payloads(pickle.loads(pickle.dumps(packed)))
    expected = torch.stack([torch.cat(layer0), torch.cat(layer1)])
    assert torch.equal(recovered["a"]["activations"]["residual_stream"], expected)
    assert torch.equal(recovered["b"]["activations"]["residual_stream_norm"], norms)
    assert all(start % 64 == 0 for *_, start, _ in packed["packed_index"])


@pytest.mark.parametrize(
    ("entries", "shape"),
    [(1, [2, 4, 8]), (20, [2, 4, 8]), (3, [2, 128, 512])],
    ids=["one-small", "many-small", "large"],
)
def test_packed_chunks_through_shared_files(
    tmp_path: Path, entries: int, shape: list[int]
) -> None:
    """Packed entries written to files travel as paths, small ones as byte ranges and large
    ones entry by entry, and unpacking removes the files."""
    tensors = [torch.randn(shape).to(torch.bfloat16) for _ in range(entries)]
    packed = pack_chunks(
        [
            (str(n), "residual_stream", t.dtype, shape, [t])
            for n, t in enumerate(tensors)
        ],
        str(tmp_path / "packed"),
    )
    assert ("packed_shards" in packed) == (shape[1] == 128)
    assert "packed_buffer" not in packed and any(tmp_path.iterdir())
    recovered = unpack_payloads(pickle.loads(pickle.dumps(packed)))
    for n, tensor in enumerate(tensors):
        assert torch.equal(recovered[str(n)]["activations"]["residual_stream"], tensor)
    assert not any(tmp_path.iterdir())


def test_packed_chunks_take_strided_one_element_slices(tmp_path: Path) -> None:
    """A column of a one-row step is a one-element slice with its parent's stride; it packs
    through the shared files and in memory alike."""
    step = torch.arange(5.0).reshape(1, 5)
    for path in (None, str(tmp_path / "packed")):
        packed = pack_chunks(
            [("a", "residual_stream_norm", step.dtype, [1, 1], [step[:, 4]])], path
        )
        recovered = unpack_payloads(pickle.loads(pickle.dumps(packed)))
        assert recovered["a"]["activations"]["residual_stream_norm"].tolist() == [[4.0]]

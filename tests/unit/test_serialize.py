"""Round-trip tests for tensor serialization helpers."""

from __future__ import annotations

import base64
import os
import pickle

import pytest
import torch

from vllm_lens._helpers._serialize import (
    _RAW_VIEW,
    PackEntry,
    decode_activations,
    deserialize_tensor,
    pack_chunks,
    serialize_activations,
    serialize_tensor,
    shared_segment,
    shared_size,
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


@pytest.mark.parametrize("insecure", [False, True], ids=["typed-off", "typed-on"])
@pytest.mark.parametrize("copy", [False, True], ids=["view", "bytes"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_packed_chunks_round_trip(
    dtype: torch.dtype, copy: bool, insecure: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through vLLM's utility-result encoding, chunks fill their tensor in flat order and each
    tensor comes back in a storage of its own bytes, with or without typed results."""
    import vllm.envs as envs
    from vllm.v1.engine import UtilityOutput
    from vllm.v1.serial_utils import (
        CUSTOM_TYPE_RAW_VIEW,
        MsgpackDecoder,
        MsgpackEncoder,
        UtilityResult,
    )

    assert _RAW_VIEW == CUSTOM_TYPE_RAW_VIEW

    monkeypatch.setattr(envs, "VLLM_ALLOW_INSECURE_SERIALIZATION", insecure)
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
            PackEntry("c", "residual_stream", dtype, [1, 0, 3], [torch.empty(0, 3)]),
        ],
        copy=copy,
    )
    frames = MsgpackEncoder().encode(
        UtilityOutput(call_id=0, result=UtilityResult(packed))
    )
    received = MsgpackDecoder(UtilityOutput).decode([bytes(f) for f in frames])
    recovered = unpack_payloads(received.result.result)
    expected = torch.stack([torch.cat(layer) for layer in layers])
    stream = recovered["a"]["activations"]["residual_stream"]
    assert stream.dtype == dtype and torch.equal(stream, expected)
    assert stream.untyped_storage().nbytes() == stream.numel() * stream.element_size()
    assert torch.equal(recovered["a"]["activations"]["residual_stream_norm"], norms)
    assert torch.equal(recovered["b"]["activations"]["residual_stream_norm"], one[None])
    assert recovered["c"]["activations"]["residual_stream"].shape == (1, 0, 3)
    if copy:
        assert pickle.loads(pickle.dumps(packed))[0] == packed[0]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_packed_chunks_through_a_memfd(dtype: torch.dtype) -> None:
    """Tensors written into a memfd come back equal, each on its own aligned bytes and
    writable, and the segment lives as long as the tensors do."""
    entries = [
        PackEntry(
            "a", "residual_stream", dtype, [2, 3, 3], [torch.randn(6, 3).to(dtype)]
        ),
        PackEntry("b", "residual_stream_norm", torch.float32, [1, 1], [torch.ones(1)]),
        PackEntry("c", "residual_stream", dtype, [1, 0, 3], [torch.empty(0, 3)]),
    ]
    size = shared_size(entries)
    segment = shared_segment(size)
    if segment is None:
        pytest.skip("no memfd on this OS")
    fd, region = segment
    os.close(fd)
    packed = pack_chunks(entries, shared=region)
    first = -(-18 * dtype.itemsize // 64) * 64
    assert packed[1] == [0, first, first + 64] and size == first + 64
    recovered = unpack_payloads(packed, region)
    del region
    stream = recovered["a"]["activations"]["residual_stream"]
    assert torch.equal(stream.reshape(6, 3), entries[0].chunks[0])
    assert stream.untyped_storage().nbytes() == stream.numel() * stream.element_size()
    stream.add_(1)
    assert torch.equal(
        recovered["b"]["activations"]["residual_stream_norm"], torch.ones(1, 1)
    )
    assert recovered["c"]["activations"]["residual_stream"].shape == (1, 0, 3)


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

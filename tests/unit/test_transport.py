"""The offline collect transport: tensors through a memfd of this process."""

from __future__ import annotations

import mmap
import os
import sys
import warnings

import pytest
import torch

from vllm_lens._helpers._transport import (
    PackEntry,
    open_segment,
    pack_chunks,
    shared_segment,
    shared_size,
    unpack_payloads,
)


@pytest.mark.skipif(sys.platform != "linux", reason="memfd is Linux only")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_round_trip(dtype: torch.dtype) -> None:
    """Chunks fill their tensor in flat order, and a float32 tensor after an odd-sized one
    unpacks because offsets are aligned."""
    layers = [
        [torch.randn(2, 3).to(dtype), torch.randn(1, 3).to(dtype)] for _ in range(2)
    ]
    one = torch.randn(1, 3)[:, 2]
    norms = torch.randn(1, 3)
    entries = [
        PackEntry("a", "residual_stream", dtype, [2, 3, 3], layers[0] + layers[1]),
        PackEntry("a", "residual_stream_norm", torch.float32, [1, 3], [norms]),
        PackEntry("b", "residual_stream_norm", torch.float32, [1, 1], [one]),
    ]
    segment, region = shared_segment(shared_size(entries))
    try:
        with open_segment(segment) as worker:
            packed = pack_chunks(entries, worker)
    finally:
        os.close(segment.fd)
    recovered = unpack_payloads(packed, region)
    stream = recovered["a"]["activations"]["residual_stream"]
    assert stream.dtype == dtype
    assert torch.equal(stream, torch.stack([torch.cat(layer) for layer in layers]))
    assert torch.equal(recovered["a"]["activations"]["residual_stream_norm"], norms)
    assert torch.equal(recovered["b"]["activations"]["residual_stream_norm"], one[None])
    assert all(offset % 64 == 0 for offset in packed[1])


@pytest.mark.parametrize("rows", [1, 3], ids=["short", "long"])
def test_a_wrong_row_count_raises(rows: int) -> None:
    """Chunks that do not fill the shape exactly raise, in place of filling another entry."""
    entry = PackEntry(
        "a", "residual_stream", torch.float32, [2, 4], [torch.ones(rows, 4)]
    )
    with pytest.raises(ValueError, match="chunks hold"):
        pack_chunks([entry], None)


def test_empty_entries_need_no_segment() -> None:
    """Empty tensors pack and unpack without a segment."""
    entry = PackEntry(
        "a", "residual_stream", torch.float32, [1, 0, 4], [torch.ones(0, 4)]
    )
    assert shared_size([entry]) == 0
    stream = unpack_payloads(pack_chunks([entry], None), None)["a"]["activations"]
    assert stream["residual_stream"].shape == (1, 0, 4)


@pytest.mark.skipif(sys.platform != "linux", reason="memfd is Linux only")
@pytest.mark.parametrize("wrong", ["name", "size"])
def test_another_memfd_is_refused(wrong: str) -> None:
    """A worker maps only the memfd it was sent: same name, same size."""
    segment, _ = shared_segment(128)
    try:
        bad = segment._replace(**{wrong: "vllm-lens-other" if wrong == "name" else 256})
        with pytest.raises(RuntimeError, match="needs each worker on the host"):
            open_segment(bad)
    finally:
        os.close(segment.fd)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_writable_mapping_round_trip(dtype: torch.dtype) -> None:
    """Writable mappings decode without warnings and keep tensors as views."""
    values = torch.arange(6).to(dtype).reshape(2, 3)
    entries = [PackEntry("a", "residual_stream", dtype, [2, 3], [values])]
    with mmap.mmap(-1, shared_size(entries)) as region:
        packed = pack_chunks(entries, region)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            recovered = unpack_payloads(packed, region)
        assert not caught
        stream = recovered["a"]["activations"]["residual_stream"]
        assert torch.equal(stream, values)
        stream.zero_()
        assert region[: values.numel() * dtype.itemsize] == bytes(
            values.numel() * dtype.itemsize
        )
        del stream, recovered

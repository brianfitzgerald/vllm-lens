"""The fused norm after an armed layer against the unfused math: vLLM's fused_add_rms_norm
on the steered stream, with the stream written to its ring rows."""

import pytest
import torch


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "steer,matched,capture",
    [(False, False, True), (True, False, False), (True, True, True)],
    ids=["capture", "steer", "steer-norm-capture"],
)
def test_lens_norm_matches_the_unfused_path(dtype, steer, matched, capture):
    from vllm_lens._fused_norm import _lens_norm

    generator = torch.Generator(device="cuda").manual_seed(0)
    n, width, eps = 37, 896, 1e-6

    def randn(*shape, kind=dtype):
        return torch.randn(*shape, generator=generator, device="cuda").to(kind)

    x, residual, weight = randn(n, width), randn(n, width), randn(width)
    ids = torch.randint(0, 3, (64,), generator=generator, device="cuda")
    table = torch.float32 if matched else dtype
    # Strided like the worker's per-layer views of its [rows, layers, hidden] tables.
    add = randn(3, 2, width, kind=table)[:, 1]
    add[0] = 0
    norm_table = (
        randn(3, 2, width, kind=torch.float32)[:, 1] * 0.01 if matched else None
    )
    ring = torch.zeros(9, width, dtype=dtype, device="cuda")
    dest = torch.full((64,), 8, dtype=torch.int64, device="cuda")
    dest[[3, 20, 36]] = torch.tensor([0, 5, 2], device="cuda")
    out, residual_out = _lens_norm(
        x,
        residual,
        weight,
        eps,
        add if steer else None,
        norm_table,
        torch.stack([ids, dest]),
        ring,
        capture,
    )

    stream = x.float() + residual.float()
    if steer:
        delta = add[ids[:n]].float()
        if norm_table is not None:
            delta = delta + norm_table[ids[:n]] * stream.norm(dim=-1, keepdim=True)
        stream = stream + delta
    expected = stream * torch.rsqrt(stream.pow(2).mean(dim=-1, keepdim=True) + eps)
    expected = (expected.to(weight.dtype) * weight).to(dtype)
    # The norm term sums in another order, so only an unmatched stream is bitwise equal.
    exact = {} if matched else {"rtol": 0, "atol": 0}
    torch.testing.assert_close(residual_out, stream.to(dtype), **exact)
    torch.testing.assert_close(out, expected)
    if capture:
        assert torch.equal(ring[[0, 5, 2]], residual_out[[3, 20, 36]])
        assert not ring[[1, 3, 4, 6, 7, 8]].any()
    else:
        assert not ring.any()

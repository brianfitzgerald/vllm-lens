"""A kernel that expands a step's per-request plan to per-token steering rows and ring rows
on the GPU, so the host writes and copies one column per request, not one per token."""

from __future__ import annotations

from vllm.triton_utils import tl, triton

# The rows of the per-request plan: steering row, ring base (-1 for none), skipped tokens,
# captured tokens and the request's first batch token.
STEER, RING, SKIP, COUNT, START = range(5)


# These counts change every step; a specialization per value class would compile again.
@triton.jit(do_not_specialize=["n_reqs", "n_tokens", "length"])
def route_kernel(
    plan,
    meta,
    plan_stride,
    meta_stride,
    n_reqs,
    n_tokens,
    length,
    sentinel,
    STEPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Token ``t`` of request ``b`` gets ``plan[STEER, b]`` and, when it is one of the
    request's captured tokens, ring row ``base + t - start - skip``; tokens past
    ``n_tokens`` get row 0 and the sentinel."""
    tokens = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inside = tokens < n_tokens
    low = tl.zeros((BLOCK,), dtype=tl.int64)
    high = tl.zeros((BLOCK,), dtype=tl.int64) + n_reqs - 1
    for _ in range(STEPS):
        middle = (low + high + 1) // 2
        first = tl.load(plan + 4 * plan_stride + middle, mask=inside, other=0)
        later = first <= tokens
        low = tl.where(later, middle, low)
        high = tl.where(later, high, middle - 1)
    steer = tl.load(plan + low, mask=inside, other=0)
    base = tl.load(plan + plan_stride + low, mask=inside, other=-1)
    skip = tl.load(plan + 2 * plan_stride + low, mask=inside, other=0)
    count = tl.load(plan + 3 * plan_stride + low, mask=inside, other=0)
    start = tl.load(plan + 4 * plan_stride + low, mask=inside, other=0)
    offset = tokens - start - skip
    captured = inside & (base >= 0) & (offset >= 0) & (offset < count)
    ring_row = tl.where(captured, base + offset, sentinel)
    in_meta = tokens < length
    tl.store(meta + tokens, tl.where(inside, steer, 0), mask=in_meta)
    tl.store(meta + meta_stride + tokens, ring_row, mask=in_meta)

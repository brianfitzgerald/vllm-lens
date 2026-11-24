"""An op at the start of the forward pass that expands a step's per-request plan to
per-token steering rows and ring rows; it runs inside the CUDA graph, so a step launches
no kernel for it."""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

# The ops live as long as this library, which is why it is held at module level.
_LIBRARY = torch.library.Library("vllm_lens", "FRAGMENT")

# The rows of the per-request plan: steering row, ring base (-1 for none), skipped tokens,
# captured tokens, and the counts of the steering and ring patches (columns 0 and 1).
STEER, RING, SKIP, COUNT, PATCHES = range(5)


@triton.jit
def _find(keys, count, value, STEPS: tl.constexpr):
    """The last index ``k < count`` of sorted ``keys`` with ``keys[k] <= value``, or -1."""
    low = tl.zeros(value.shape, dtype=tl.int64)
    high = tl.zeros(value.shape, dtype=tl.int64) + count
    for _ in range(STEPS):
        active = low < high
        middle = (low + high) // 2
        key = tl.load(keys + middle, mask=active, other=0).to(tl.int64)
        below = active & (key <= value)
        low = tl.where(below, middle + 1, low)
        high = tl.where(active & ~below, middle, high)
    return low - 1


# The token count changes with the batch; a specialization per value class would compile again.
@triton.jit(do_not_specialize=["n_tokens"])
def route_kernel(
    plan,
    starts,
    meta,
    patches,
    plan_stride,
    meta_stride,
    patch_stride,
    n_reqs,
    n_tokens,
    sentinel,
    STEPS: tl.constexpr,
    PATCH_STEPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Token ``t`` of batch row ``b`` (``starts`` is the runner's ``query_start_loc``) gets
    ``plan[STEER, b]`` and, when it is one of the row's captured tokens, ring row
    ``base + t - start - skip``; a patch for ``t`` replaces either. Tokens past the batch
    get row 0 and the sentinel."""
    # The plan rows are STEER, RING, SKIP, COUNT and PATCHES, in that order.
    tokens = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inside = tokens < n_tokens
    real = inside & (tokens < tl.load(starts + n_reqs).to(tl.int64))
    row = tl.maximum(_find(starts, n_reqs, tokens, STEPS), 0)
    steer = tl.load(plan + 0 * plan_stride + row, mask=real, other=0)
    base = tl.load(plan + 1 * plan_stride + row, mask=real, other=-1)
    skip = tl.load(plan + 2 * plan_stride + row, mask=real, other=0)
    count = tl.load(plan + 3 * plan_stride + row, mask=real, other=0)
    start = tl.load(starts + row, mask=real, other=0).to(tl.int64)
    offset = tokens - start - skip
    captured = real & (base >= 0) & (offset >= 0) & (offset < count)
    ring_row = tl.where(captured, base + offset, sentinel)
    for kind in tl.static_range(2):
        keys = patches + 2 * kind * patch_stride
        found = _find(keys, tl.load(plan + 4 * plan_stride + kind), tokens, PATCH_STEPS)
        hit = real & (found >= 0)
        hit &= tl.load(keys + found, mask=hit, other=-1) == tokens
        value = tl.load(keys + patch_stride + found, mask=hit, other=0)
        if kind == 0:
            steer = tl.where(hit, value, steer)
        else:
            ring_row = tl.where(hit, value, ring_row)
    tl.store(meta + tokens, tl.where(real, steer, 0), mask=inside)
    tl.store(meta + meta_stride + tokens, ring_row, mask=inside)


def _route(
    plan: torch.Tensor,
    starts: torch.Tensor,
    meta: torch.Tensor,
    patches: torch.Tensor,
    n_tokens: int,
    sentinel: int,
) -> None:
    """Write the steering row and ring row of each of the first ``n_tokens`` tokens to
    ``meta``; ``patches[kind]`` holds sorted tokens and their values."""
    if n_tokens == 0:
        return
    n_reqs = starts.shape[0] - 1
    route_kernel[(triton.cdiv(n_tokens, 256),)](  # type: ignore[reportIndexIssue]
        plan,
        starts,
        meta,
        patches,
        plan.stride(0),
        meta.stride(0),
        patches.stride(1),
        n_reqs,
        n_tokens,
        sentinel,
        STEPS=max(1, n_reqs.bit_length()),
        PATCH_STEPS=max(1, patches.shape[-1].bit_length()),
        BLOCK=256,
    )


direct_register_custom_op("route", _route, mutates_args=["meta"], target_lib=_LIBRARY)

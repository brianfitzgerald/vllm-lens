"""The route op against a per-token reference: steering rows for every token, ring rows for
captured tokens, patches, and row 0 and the sentinel for padding tokens."""

import pytest
import torch


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("padding", [0, 9])
def test_route_matches_the_per_token_plan(seed, padding):
    from vllm_lens._route import COUNT, PATCHES, RING, SKIP, _route

    generator = torch.Generator().manual_seed(seed)
    n_reqs, max_reqs, max_tokens, sentinel = 6, 16, 128, 99
    counts = torch.randint(2, 9, (n_reqs,), generator=generator).tolist()
    bounds = [0]
    for count in counts:
        bounds.append(bounds[-1] + count)
    n = bounds[-1]
    # The runner's query_start_loc: rows past the batch start at the token count.
    starts = torch.full((max_reqs + 1,), n, dtype=torch.int32)
    starts[: n_reqs + 1] = torch.tensor(bounds)
    # Each runner index's steering row, and a shuffled runner index per batch row; stale
    # rows past the batch must not reach any token.
    rows = torch.randint(0, 13, (max_reqs,), generator=generator)
    batch_idx = torch.randperm(max_reqs, generator=generator)
    plan = torch.zeros((4, max_reqs), dtype=torch.int64)
    plan[RING] = 5
    plan[RING, :n_reqs] = -1
    plan[COUNT] = 50
    plan[COUNT, :n_reqs] = 0
    # A run from the request's start, one skipped-to token, and a shortened run.
    plan[RING, 0], plan[COUNT, 0] = 10, counts[0]
    plan[RING, 2], plan[SKIP, 2], plan[COUNT, 2] = 30, counts[2] - 1, 1
    plan[RING, 4], plan[COUNT, 4] = 50, counts[4] - 1
    # Steering patches on request 1's tokens, ring patches on request 3's tokens.
    patches = torch.full((2, 2, max_tokens), -5, dtype=torch.int64)
    steered = list(range(bounds[1], bounds[2]))[::2]
    rowed = list(range(bounds[3], bounds[4]))[1:]
    patches[0, 0, : len(steered)] = torch.tensor(steered)
    patches[0, 1, : len(steered)] = 40
    patches[1, 0, : len(rowed)] = torch.tensor(rowed)
    patches[1, 1, : len(rowed)] = torch.arange(60, 60 + len(rowed))
    plan[PATCHES, 0], plan[PATCHES, 1] = len(steered), len(rowed)
    total = n + padding
    expected = torch.full((2, total + 5), -7, dtype=torch.int64)
    expected[0, :total] = 0
    expected[1, :total] = sentinel
    for b in range(n_reqs):
        for t in range(bounds[b], bounds[b + 1]):
            expected[0, t] = rows[batch_idx[b]]
            offset = t - bounds[b] - int(plan[SKIP, b])
            if plan[RING, b] >= 0 and 0 <= offset < plan[COUNT, b]:
                expected[1, t] = plan[RING, b] + offset
    expected[0, steered] = 40
    expected[1, rowed] = torch.arange(60, 60 + len(rowed))
    meta = torch.full((2, total + 5), -7, dtype=torch.int64, device="cuda")
    _route(
        plan.cuda(),
        rows.cuda(),
        batch_idx.cuda(),
        starts.cuda(),
        meta,
        patches.cuda(),
        total,
        sentinel,
    )
    assert torch.equal(meta.cpu(), expected)


def test_route_does_not_compile_again_for_new_counts():
    """A launch with a token count of another value class (1, a multiple of 16, other)
    reuses the compiled kernel, so a step never waits for Triton."""
    import time

    from vllm_lens._route import _route

    plan = torch.zeros((4, 64), dtype=torch.int64, device="cuda")
    rows = torch.zeros(64, dtype=torch.int64, device="cuda")
    starts = torch.zeros(65, dtype=torch.int32, device="cuda")
    meta = torch.zeros((2, 512), dtype=torch.int64, device="cuda")
    patches = torch.zeros((2, 2, 512), dtype=torch.int64, device="cuda")

    def launch(n_tokens: int) -> float:
        start = time.perf_counter()
        _route(plan, rows, rows, starts, meta, patches, n_tokens, 99)
        torch.cuda.synchronize()
        return time.perf_counter() - start

    launch(37)
    seconds = [launch(t) for t in (1, 32, 48, 320, 7)]
    assert max(seconds) < 0.02, seconds

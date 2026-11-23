"""The route kernel against the per-token plan it replaces: steering rows for every token,
ring rows for captured tokens, and row 0 and the sentinel past the batch."""

import pytest
import torch


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_route_matches_the_per_token_plan(seed):
    from vllm_lens._route import COUNT, RING, SKIP, START, STEER, route_kernel

    generator = torch.Generator().manual_seed(seed)
    n_reqs, max_reqs, sentinel = 6, 16, 99
    counts = torch.randint(1, 9, (n_reqs,), generator=generator).tolist()
    bounds = [0]
    for count in counts:
        bounds.append(bounds[-1] + count)
    n, length = bounds[-1], bounds[-1] + 7
    plan = torch.zeros((5, max_reqs), dtype=torch.int64)
    plan[STEER, :n_reqs] = torch.randint(0, 13, (n_reqs,), generator=generator)
    plan[RING, :n_reqs] = -1
    plan[START, :n_reqs] = torch.tensor(bounds[:-1])
    # A run from the request's start, one skipped-to token, and a shortened run.
    plan[RING, 0], plan[COUNT, 0] = 10, counts[0]
    plan[RING, 2], plan[SKIP, 2], plan[COUNT, 2] = 30, counts[2] - 1, 1
    plan[RING, 4], plan[COUNT, 4] = 50, max(1, counts[4] - 2)
    expected = torch.zeros((2, length + 5), dtype=torch.int64)
    expected[0, length:] = -7
    expected[1, :length] = sentinel
    expected[1, length:] = -7
    for b in range(n_reqs):
        for t in range(bounds[b], bounds[b + 1]):
            expected[0, t] = plan[STEER, b]
            offset = t - bounds[b] - int(plan[SKIP, b])
            if plan[RING, b] >= 0 and 0 <= offset < plan[COUNT, b]:
                expected[1, t] = plan[RING, b] + offset
    meta = torch.full((2, length + 5), -7, dtype=torch.int64, device="cuda")
    table = plan.cuda()
    route_kernel[(-(-length // 256),)](  # type: ignore[reportIndexIssue]
        table,
        meta,
        table.stride(0),
        meta.stride(0),
        n_reqs,
        n,
        length,
        sentinel,
        STEPS=max_reqs.bit_length(),
        BLOCK=256,
    )
    assert torch.equal(meta.cpu(), expected)


def test_route_does_not_compile_again_for_new_counts():
    """A launch with counts of another value class (1, a multiple of 16, other) reuses the
    compiled kernel, so a step never waits for Triton."""
    import time

    from vllm_lens._route import route_kernel

    plan = torch.zeros((5, 64), dtype=torch.int64, device="cuda")
    meta = torch.zeros((2, 512), dtype=torch.int64, device="cuda")

    def launch(n_reqs: int, n_tokens: int) -> float:
        start = time.perf_counter()
        route_kernel[(-(-n_tokens // 256),)](  # type: ignore[reportIndexIssue]
            plan,
            meta,
            plan.stride(0),
            meta.stride(0),
            n_reqs,
            n_tokens,
            n_tokens,
            99,
            STEPS=7,
            BLOCK=256,
        )
        torch.cuda.synchronize()
        return time.perf_counter() - start

    launch(3, 37)
    seconds = [launch(n, t) for n, t in ((1, 1), (16, 32), (17, 48), (64, 320), (5, 7))]
    assert max(seconds) < 0.02, seconds

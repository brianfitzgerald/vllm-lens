"""The batched eager steering update gives what the per-request path gives."""

from types import SimpleNamespace

import pytest
import torch

from vllm_lens._helpers.types import SteeringVector
from vllm_lens._worker_ext import (
    _apply_steering,
    _apply_steering_rows,
    _eager_steer_row,
)

HIDDEN = 16
LAYER = 3


@pytest.mark.parametrize("norm_match", [False, True])
@pytest.mark.parametrize("fused", [False, True], ids=["plain", "fused-residual"])
def test_batched_rows_match_the_per_request_path(norm_match: bool, fused: bool) -> None:
    """Three requests with their own vectors, one of them unsteered, end up as the
    per-request path leaves them, to float32 rounding."""
    generator = torch.Generator().manual_seed(0)
    output = torch.randn(10, HIDDEN, generator=generator)
    residual = torch.randn(10, HIDDEN, generator=generator) if fused else None
    norm_ref = output + residual if residual is not None else output
    spans = [(0, 4), (6, 10)]
    configs = [
        SteeringVector(
            activations=torch.randn(1, HIDDEN, generator=generator),
            layer_indices=[LAYER],
            scale=scale,
            norm_match=norm_match,
        )
        for scale in (0.5, -2.0)
    ]
    expected = output.clone()
    for (start, end), config in zip(spans, configs):
        _apply_steering([config], LAYER, expected, start, end, 0, norm_ref)
    extension = SimpleNamespace(_eager_rows={})
    rows = [
        _eager_steer_row(extension, f"r{n}", config, LAYER)  # type: ignore[arg-type]
        for n, config in enumerate(configs)
    ]
    batched = output.clone()
    _apply_steering_rows(rows, spans, batched, norm_ref)
    assert torch.allclose(batched, expected, atol=1e-4)
    assert torch.equal(batched[4:6], output[4:6])
    assert set(extension._eager_rows) == {"r0", "r1"}

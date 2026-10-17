"""Steering lookup, the per-request row table, and the paths that apply it. No GPU."""

from types import SimpleNamespace

import pytest
import torch

from vllm_lens._helpers.types import SteeringVector
from vllm_lens._worker_ext import (
    _apply_steering,
    _apply_table_rows,
    _request_hooks,
    _steering_key,
    _SteerTable,
)

HIDDEN = 8
LAYER = 2
MAX_TOKENS = 16


def _vector(seed: int, **kwargs) -> SteeringVector:
    """A reproducible random vector on ``LAYER``."""
    generator = torch.Generator().manual_seed(seed)
    return SteeringVector(
        activations=torch.randn(1, HIDDEN, generator=generator),
        layer_indices=[LAYER],
        **kwargs,
    )


@pytest.mark.parametrize(
    ("internal_id", "extra_args", "expected"),
    [
        ("0", {"_steering_id": "_steer_3"}, "_steer_3"),
        ("0", {"_steering_id": "_steer_9"}, None),
        ("cmpl-abc-0", None, "cmpl-abc"),
        ("cmpl-abc-0", {}, "cmpl-abc"),
        ("cmpl-abc-x-0", None, None),
        ("other-7", None, None),
    ],
    ids=["offline", "offline-unknown", "async-suffix", "async-empty", "dashed", "miss"],
)
def test_lookup(internal_id: str, extra_args: dict | None, expected: str | None):
    """Offline ids resolve through the sentinel, async ids through the exact external id."""
    extension = SimpleNamespace(
        _steering_data={"_steer_3": [], "cmpl-abc": []},
        _hook_data={"_steer_3": ["offline"], "cmpl-abc": ["async"]},
    )
    assert _steering_key(extension, internal_id, extra_args) == expected  # type: ignore[arg-type]
    hook_args = extra_args and {"_hook_id": extra_args["_steering_id"]}
    hooks = _request_hooks(extension, internal_id, hook_args)  # type: ignore[arg-type]
    assert hooks == (extension._hook_data[expected] if expected else [])


def test_table_rows_are_reused_and_grown():
    """Released rows are reused, the table doubles when full, and old rows survive growth."""
    table = _SteerTable(device=torch.device("cpu"), hidden=HIDDEN, capacity=2)
    first = _vector(0, scale=2.0)
    assert table.row("a", [first]) == 1
    assert table.row("a", [first]) == 1
    assert table.row("b", [_vector(1)]) == 2
    assert table.capacity == 4
    torch.testing.assert_close(table.add[LAYER][1], first.activations[0] * 2.0)
    table.release("a")
    assert table.row("c", [_vector(2)]) == 1
    assert table.row("d", [_vector(3)]) == 3


@pytest.mark.parametrize(
    ("activations", "expected"),
    [(torch.ones(1, HIDDEN + 1), 0), (torch.ones(1, 2, HIDDEN), -1)],
    ids=["wrong-width", "position-specific"],
)
def test_a_vector_outside_the_table(activations: torch.Tensor, expected: int):
    """A bad vector gets row 0 (no steering); a position-specific one gets -1."""
    table = _SteerTable(device=torch.device("cpu"), hidden=HIDDEN)
    vector = SteeringVector(activations=activations, layer_indices=[LAYER])
    assert table.row("a", [vector]) == expected
    # A failed row goes back to the free list; no row stays taken.
    assert table.row("b", [_vector(0)]) == 1


@pytest.mark.parametrize("norm_match", [False, True])
@pytest.mark.parametrize("fused", [False, True], ids=["plain", "fused-residual"])
def test_table_rows_match_the_per_request_path(norm_match: bool, fused: bool):
    """Two steered requests around an unsteered one end up as the per-request path leaves
    them, to float32 rounding."""
    generator = torch.Generator().manual_seed(0)
    output = torch.randn(10, HIDDEN, generator=generator)
    residual = torch.randn(10, HIDDEN, generator=generator) if fused else None
    norm_ref = output + residual if residual is not None else output
    spans = [(0, 4), (6, 10)]
    configs = [
        _vector(n, scale=scale, norm_match=norm_match)
        for n, scale in [(1, 0.5), (2, -2.0)]
    ]
    expected = output.clone()
    for (start, end), config in zip(spans, configs):
        _apply_steering([config], LAYER, expected, start, end, 0, norm_ref)
    table = _SteerTable(device=torch.device("cpu"), hidden=HIDDEN)
    rows = [table.row(f"r{n}", [config]) for n, config in enumerate(configs)]
    batched = output.clone()
    _apply_table_rows(
        table,
        LAYER,
        [(row, *span) for row, span in zip(rows, spans)],
        batched,
        norm_ref,
    )
    torch.testing.assert_close(batched, expected, atol=1e-4, rtol=1e-4)
    assert torch.equal(batched[4:6], output[4:6])


def test_clearing_a_request_releases_its_table_rows():
    """``clear_request_state`` gives a key's table row back and forgets a position-specific
    key."""
    from vllm_lens._worker_ext import HiddenStatesExtension

    table = _SteerTable(device=torch.device("cpu"), hidden=HIDDEN)
    positional = SteeringVector(
        activations=torch.ones(1, 2, HIDDEN), layer_indices=[LAYER]
    )
    data = {"a": [_vector(0)], "p": [positional]}
    assert table.row("a", data["a"]) == 1 and table.row("p", data["p"]) == -1
    worker = HiddenStatesExtension.__new__(HiddenStatesExtension)
    worker._steering_data, worker._hook_data, worker._steer_table = data, {}, table
    worker.clear_request_state(["a", "p"], [], [])
    assert worker._steering_data == {} and table.free == [1]
    assert table.rows == {}

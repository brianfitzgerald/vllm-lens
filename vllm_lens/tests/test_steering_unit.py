"""The steering row table and the per-pass row index. No GPU."""

import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm_lens._worker_ext as worker_ext
from vllm_lens import SteeringVector
from vllm_lens._worker_ext import (
    HiddenStatesExtension,
    _steer,
    _steer_pass,
    _SteerTable,
    norm_match,
)

HIDDEN = 8
LAYER = 2


def _table(n_rows: int = 4) -> _SteerTable:
    """A CPU table of ``n_rows`` rows."""
    return _SteerTable(device=torch.device("cpu"), hidden=HIDDEN, n_rows=n_rows)


def _vector(seed: int, positions: int | None = None, **kwargs) -> SteeringVector:
    """A reproducible random vector on ``LAYER``; 3-D with ``positions`` rows."""
    generator = torch.Generator().manual_seed(seed)
    shape = (1, HIDDEN) if positions is None else (1, positions, HIDDEN)
    return SteeringVector(
        activations=torch.randn(shape, generator=generator),
        layer_indices=[LAYER],
        **kwargs,
    )


def test_rows_are_reused_and_run_out(caplog: pytest.LogCaptureFixture) -> None:
    """Released rows are reused; with no free row a request gets row 0 and a warning."""
    table = _table(n_rows=3)
    first = _vector(0, scale=2.0)
    assert table.row("a", [first]) == 1
    assert table.row("a", [first]) == 1
    torch.testing.assert_close(table.add[LAYER][1], first.activations[0] * 2.0)
    assert table.row("b", [_vector(1)]) == 2
    assert table.row("c", [_vector(2)]) == 0
    assert "steering rows are in use" in caplog.text
    table.release("a")
    assert table.row("d", [_vector(3)]) == 1
    assert table.add[LAYER].shape == (3 + worker_ext.SCRATCH_ROWS, HIDDEN)


@pytest.mark.parametrize(
    ("vectors", "row", "positional"),
    [
        ([_vector(0)], 1, 0),
        ([_vector(0, positions=2)], 0, 1),
        ([_vector(0), _vector(1, positions=2)], 1, 1),
    ],
    ids=["flat", "positional", "both"],
)
def test_a_request_row_holds_its_flat_vectors(
    vectors: list[SteeringVector], row: int, positional: int
) -> None:
    """2-D vectors share a row; 3-D vectors are kept for the scratch rows."""
    table = _table()
    assert table.row("a", vectors) == row
    assert len(table.positional.get("a", [])) == positional
    assert table.row("b", [_vector(5)]) == row + 1


@pytest.mark.parametrize(
    ("width", "layer", "error"),
    [(HIDDEN, 9, "out of range"), (HIDDEN + 1, LAYER, "vector width")],
    ids=["layer", "width"],
)
def test_bad_vectors_are_rejected(width: int, layer: int, error: str) -> None:
    """A vector on a missing layer or of the wrong width fails the RPC; a reused key gives
    its row back."""
    extension = SimpleNamespace(
        model_runner=SimpleNamespace(
            model=torch.nn.Linear(1, 1),
            model_config=SimpleNamespace(get_hidden_size=lambda: HIDDEN),
        ),
        model_config=SimpleNamespace(get_total_num_hidden_layers=lambda: 4),
        _steering_data={},
        _steer_table=_table(),
    )
    HiddenStatesExtension.set_steering_data(extension, "a", pickle.dumps([_vector(0)]))  # type: ignore[arg-type]
    assert extension._steer_table.row("a", extension._steering_data["a"]) == 1
    bad = SteeringVector(activations=torch.ones(1, width), layer_indices=[layer])
    with pytest.raises(ValueError, match=error):
        HiddenStatesExtension.set_steering_data(extension, "a", pickle.dumps([bad]))  # type: ignore[arg-type]
    HiddenStatesExtension.set_steering_data(extension, "a", pickle.dumps([_vector(1)]))  # type: ignore[arg-type]
    assert "a" not in extension._steer_table.rows
    assert extension._steer_table.free == [3, 2, 1]


def _plan(
    monkeypatch: pytest.MonkeyPatch,
    payloads: list[list[SteeringVector] | None],
    lengths: list[int],
    computed: list[int],
) -> tuple[SimpleNamespace, worker_ext._SteerPass | None]:
    """Run _steer_pass over one step with one request per payload, ``lengths`` tokens each
    in this step and ``computed`` tokens before it."""
    requests = {
        f"r{i}-0": SimpleNamespace(
            sampling_params=SimpleNamespace(extra_args={}), num_computed_tokens=done
        )
        for i, done in enumerate(computed)
    }
    bounds = np.concatenate([[0], np.cumsum(lengths)])
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(num_reqs=len(payloads), req_ids=list(requests)),
        requests=requests,
    )
    extension = SimpleNamespace(
        model_runner=runner,
        _steering_data={
            f"r{i}": vectors for i, vectors in enumerate(payloads) if vectors
        },
        _steer_table=_table(1 + len(payloads)),
    )
    monkeypatch.setattr(
        worker_ext, "_batch_layout", lambda runner, warn: torch.from_numpy(bounds)
    )
    return extension, _steer_pass(extension)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("payloads", "rows"),
    [([[_vector(0)], None, [_vector(1)]], [1, 1, 1, 0, 0, 2]), ([None, None], None)],
    ids=["steered", "unsteered"],
)
def test_tokens_take_their_request_row(
    monkeypatch: pytest.MonkeyPatch, payloads: list, rows: list[int] | None
) -> None:
    """Steered requests' tokens take their row and an unsteered request's take row 0; a
    step without steering has no plan."""
    _, plan = _plan(monkeypatch, payloads, [3, 2, 1][: len(payloads)], [0, 0, 5])
    if rows is None:
        assert plan is None
    else:
        assert plan is not None and plan.layers == {LAYER}
        assert plan.rows.tolist() == rows


@pytest.mark.parametrize("norm", [False, True], ids=["add", "norm-match"])
def test_positions_take_scratch_rows_across_chunks(
    monkeypatch: pytest.MonkeyPatch, norm: bool
) -> None:
    """A 3-D vector steers its absolute positions in this chunk through scratch rows that
    also hold the request's 2-D vectors."""
    flat = _vector(0, norm_match=norm)
    positional = _vector(1, positions=6, position_indices=[1, 4, 5, 9, 2, 3])
    # Positions 4 and 5 fall in this chunk of 3 tokens (positions 3..5).
    extension, plan = _plan(monkeypatch, [[flat, positional]], [3], [3])
    assert plan is not None
    table = extension._steer_table
    scratch = table.n_rows
    assert plan.rows.tolist() == [scratch, scratch + 1, scratch + 2]
    tables = table.norm if norm else table.add
    base = tables[LAYER][1]
    for slot, pi in [(0, 5), (1, 1), (2, 2)]:
        torch.testing.assert_close(
            table.add[LAYER][scratch + slot] - (0 if norm else base),
            positional.activations[0, pi].float(),
        )
        if norm:
            torch.testing.assert_close(table.norm[LAYER][scratch + slot], base)


def test_positions_over_the_scratch_rows_are_skipped(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Positions past ``SCRATCH_ROWS`` in one step keep the request row, with a warning."""
    monkeypatch.setattr(worker_ext, "SCRATCH_ROWS", 2)
    extension, plan = _plan(monkeypatch, [[_vector(1, positions=4)]], [4], [0])
    assert plan is not None
    scratch = extension._steer_table.n_rows
    assert plan.rows.tolist() == [scratch, scratch + 1, 0, 0]
    assert "over the 2 scratch rows" in caplog.text


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("fused", [False, True], ids=["plain", "fused-residual"])
@pytest.mark.parametrize("norm", [False, True], ids=["add", "norm-match"])
def test_steer_matches_the_per_request_sum(
    dtype: torch.dtype, fused: bool, norm: bool
) -> None:
    """Two steered requests around an unsteered one end up as the per-request sum leaves
    them, with ``norm_match`` scaled by the unsteered stream."""
    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(10, HIDDEN, generator=generator).to(dtype)
    residual = torch.randn(10, HIDDEN, generator=generator).to(dtype) if fused else None
    stream = hidden.float() + (residual.float() if residual is not None else 0)
    table = _table()
    spans = [(0, 4), (6, 10)]
    vectors = [_vector(n, scale=s, norm_match=norm) for n, s in [(1, 0.5), (2, -2.0)]]
    index = torch.zeros(10, dtype=torch.int32)
    expected = hidden.float().clone()
    for (start, end), vector in zip(spans, vectors):
        index[start:end] = table.row(str(start), [vector])
        v = vector.activations[0].float()
        delta = norm_match(stream[start:end], v.expand(end - start, -1)) if norm else v
        expected[start:end] += delta * vector.scale
    steered = hidden.clone()
    add, norm_rows = table.layer(LAYER)
    _steer(steered, residual, add, norm_rows, index)
    tolerance = 1e-5 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(
        steered.float(), expected, atol=tolerance, rtol=tolerance
    )
    assert torch.equal(steered[4:6], hidden[4:6])

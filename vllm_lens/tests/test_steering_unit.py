"""Request payloads, the steering row table, and the per-pass row index. No GPU."""

import pickle
from types import SimpleNamespace

import cloudpickle
import numpy as np
import pytest
import torch

import vllm_lens._worker_ext as worker_ext
from vllm_lens import Hook, SteeringVector
from vllm_lens._helpers.types import HOOKS_PAYLOAD, STEERING_PAYLOAD
from vllm_lens._worker_ext import (
    _release_payloads,
    _request_payload,
    _steer,
    _steer_pass,
    _SteerTable,
    norm_match,
)

HIDDEN = 8
LAYER = 2


def _extension(table: _SteerTable | None = None) -> SimpleNamespace:
    """An extension with a float16 CPU model."""
    model = torch.nn.Linear(1, 1, dtype=torch.float16)
    return SimpleNamespace(
        model_runner=SimpleNamespace(model=model),
        _payloads={},
        _request_payloads={},
        _steer_table=table,
    )


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


def _state(extra: dict | None) -> SimpleNamespace:
    """A runner request state with ``extra`` as its extra_args."""
    return SimpleNamespace(sampling_params=SimpleNamespace(extra_args=extra))


@pytest.mark.parametrize("key", [STEERING_PAYLOAD, HOOKS_PAYLOAD])
def test_requests_share_one_decoded_payload(key: str) -> None:
    """Two requests with the same bytes share one copy, freed when both have left."""
    if key == STEERING_PAYLOAD:
        raw = pickle.dumps(
            [SteeringVector(activations=torch.ones(1, 4), layer_indices=[0])]
        )
    else:
        raw = cloudpickle.dumps([Hook(fn=lambda ctx, h: None, layer_indices=[0])])
    extension = _extension()
    first = _request_payload(extension, _state({key: raw}), "a-0", key)  # type: ignore[arg-type]
    second = _request_payload(extension, _state({key: raw}), "b-0", key)  # type: ignore[arg-type]
    assert first is second and len(first) == 1
    if key == STEERING_PAYLOAD:
        assert first[0].activations.dtype == torch.float16
    _release_payloads(extension, "a-0")  # type: ignore[arg-type]
    assert len(extension._payloads) == 1
    _release_payloads(extension, "b-0")  # type: ignore[arg-type]
    assert extension._payloads == {} and extension._request_payloads == {}


@pytest.mark.parametrize(
    "state",
    [None, _state(None), _state({}), _state({STEERING_PAYLOAD: b"not a pickle"})],
    ids=["no-state", "no-extra", "empty", "bad-bytes"],
)
def test_missing_or_bad_payload_steers_nothing(state: SimpleNamespace | None) -> None:
    """A request without a payload, or with one that cannot be decoded, gets []."""
    extension = _extension()
    assert _request_payload(extension, state, "a-0", STEERING_PAYLOAD) == []  # type: ignore[arg-type]
    _release_payloads(extension, "a-0")  # type: ignore[arg-type]
    assert extension._payloads == {}


def test_the_last_request_of_a_payload_frees_its_row() -> None:
    """A payload's row is given back when its last request leaves the runner."""
    table = _table()
    extension = _extension(table)
    state = _state({STEERING_PAYLOAD: pickle.dumps([_vector(0)])})
    for req_id in ("a-0", "b-0"):
        configs = _request_payload(extension, state, req_id, STEERING_PAYLOAD)  # type: ignore[arg-type]
        digest = extension._request_payloads[(STEERING_PAYLOAD, req_id)]
        assert table.row(digest, configs) == 1
    _release_payloads(extension, "a-0")  # type: ignore[arg-type]
    assert 1 not in table.free
    _release_payloads(extension, "b-0")  # type: ignore[arg-type]
    assert 1 in table.free and table.rows == {}


def test_rows_are_reused_and_run_out(caplog: pytest.LogCaptureFixture) -> None:
    """Released rows are reused; with no free row a payload gets row 0 and a warning."""
    table = _table(n_rows=3)
    first = _vector(0, scale=2.0)
    assert table.row(b"a", [first]) == 1
    assert table.row(b"a", [first]) == 1
    torch.testing.assert_close(table.add[LAYER][1], first.activations[0] * 2.0)
    assert table.row(b"b", [_vector(1)]) == 2
    assert table.row(b"c", [_vector(2)]) == 0
    assert "steering rows are in use" in caplog.text
    table.release(b"a")
    assert table.row(b"d", [_vector(3)]) == 1
    assert table.add[LAYER].shape == (3 + worker_ext.SCRATCH_ROWS, HIDDEN)


@pytest.mark.parametrize(
    ("vectors", "row", "positional"),
    [
        ([_vector(0)], 1, 0),
        ([_vector(0, positions=2)], 0, 1),
        ([_vector(0), _vector(1, positions=2)], 1, 1),
        (
            [
                SteeringVector(
                    activations=torch.ones(1, HIDDEN + 1), layer_indices=[LAYER]
                )
            ],
            0,
            0,
        ),
    ],
    ids=["flat", "positional", "both", "wrong-width"],
)
def test_a_payload_row_holds_its_flat_vectors(
    vectors: list[SteeringVector], row: int, positional: int
) -> None:
    """2-D vectors share a row; 3-D vectors are kept for the scratch rows; a bad one steers
    nothing and holds no row."""
    table = _table()
    assert table.row(b"a", vectors) == row
    assert len(table.positional.get(b"a", [])) == positional
    assert table.row(b"b", [_vector(5)]) == row + 1


def _runner(
    payloads: list[list[SteeringVector] | None],
    lengths: list[int],
    computed: list[int],
) -> SimpleNamespace:
    """A runner with one request per payload, ``lengths`` tokens each in this step, and
    ``computed`` tokens before it."""
    requests = {
        f"r{i}-0": SimpleNamespace(
            sampling_params=SimpleNamespace(
                extra_args={}
                if vectors is None
                else {STEERING_PAYLOAD: pickle.dumps(vectors)}
            ),
            num_computed_tokens=done,
        )
        for i, (vectors, done) in enumerate(zip(payloads, computed))
    }
    bounds = np.concatenate([[0], np.cumsum(lengths)])
    return SimpleNamespace(
        model=torch.nn.Linear(1, 1),
        input_batch=SimpleNamespace(num_reqs=len(payloads), req_ids=list(requests)),
        requests=requests,
        query_start_loc=SimpleNamespace(gpu=None, np=bounds),
        _bounds=torch.from_numpy(bounds),
    )


def _plan(
    monkeypatch: pytest.MonkeyPatch,
    payloads: list[list[SteeringVector] | None],
    lengths: list[int],
    computed: list[int],
) -> tuple[SimpleNamespace, worker_ext._SteerPass | None]:
    """Run _steer_pass over one step."""
    table = _table(n_rows=1 + len(payloads))
    runner = _runner(payloads, lengths, computed)
    extension = _extension(table)
    extension.model_runner = runner
    monkeypatch.setattr(
        worker_ext, "_batch_layout", lambda runner, warn: runner._bounds
    )
    return extension, _steer_pass(extension)  # type: ignore[arg-type]


def test_tokens_take_their_request_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """Steered requests' tokens take their row; an unsteered request's take row 0."""
    extension, plan = _plan(
        monkeypatch, [[_vector(0)], None, [_vector(1)]], [3, 2, 1], [0, 0, 5]
    )
    assert plan is not None and plan.layers == {LAYER}
    assert plan.rows.tolist() == [1, 1, 1, 0, 0, 2]


def test_no_steered_request_plans_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A step without steering returns no plan."""
    assert _plan(monkeypatch, [None, None], [2, 2], [0, 0])[1] is None


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
        index[start:end] = table.row(bytes([start]), [vector])
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

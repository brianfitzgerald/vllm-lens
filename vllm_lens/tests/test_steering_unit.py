"""Request payloads, the per-payload row table, and the paths that apply it. No GPU."""

import pickle
from types import SimpleNamespace

import cloudpickle
import pytest
import torch

from vllm_lens import Hook, SteeringVector
from vllm_lens._helpers.types import HOOKS_PAYLOAD, STEERING_PAYLOAD
from vllm_lens._worker_ext import (
    _apply_steering,
    _apply_table_rows,
    _release_payloads,
    _request_payload,
    _SteerTable,
    _token_layout,
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


def _state(extra: dict | None) -> SimpleNamespace:
    """A runner request state with ``extra`` as its extra_args."""
    return SimpleNamespace(sampling_params=SimpleNamespace(extra_args=extra))


def _vector(seed: int, **kwargs) -> SteeringVector:
    """A reproducible random vector on ``LAYER``."""
    generator = torch.Generator().manual_seed(seed)
    return SteeringVector(
        activations=torch.randn(1, HIDDEN, generator=generator),
        layer_indices=[LAYER],
        **kwargs,
    )


@pytest.mark.parametrize("key", [STEERING_PAYLOAD, HOOKS_PAYLOAD])
def test_requests_share_one_decoded_payload(key: str) -> None:
    """Two requests with the same bytes share one copy, freed when both have left."""
    if key == STEERING_PAYLOAD:
        raw = pickle.dumps([_vector(0)])
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
    """A payload's table row is given back when its last request leaves the runner."""
    table = _SteerTable(device=torch.device("cpu"), hidden=HIDDEN)
    extension = _extension(table)
    state = _state({STEERING_PAYLOAD: pickle.dumps([_vector(0)])})
    for req_id in ("a-0", "b-0"):
        configs = _request_payload(extension, state, req_id, STEERING_PAYLOAD)  # type: ignore[arg-type]
        digest = extension._request_payloads[(STEERING_PAYLOAD, req_id)]
        assert table.row(digest, configs) == 1
    _release_payloads(extension, "a-0")  # type: ignore[arg-type]
    assert table.free == []
    _release_payloads(extension, "b-0")  # type: ignore[arg-type]
    assert table.free == [1] and table.rows == {}


def test_table_rows_are_reused_and_grown():
    """Released rows are reused, the table doubles when full, and old rows survive growth."""
    table = _SteerTable(device=torch.device("cpu"), hidden=HIDDEN, capacity=2)
    first = _vector(0, scale=2.0)
    assert table.row(b"a", [first]) == 1
    assert table.row(b"a", [first]) == 1
    assert table.row(b"b", [_vector(1)]) == 2
    assert table.capacity == 4
    torch.testing.assert_close(table.add[LAYER][1], first.activations[0] * 2.0)
    table.release(b"a")
    assert table.row(b"c", [_vector(2)]) == 1
    assert table.row(b"d", [_vector(3)]) == 3


@pytest.mark.parametrize(
    ("activations", "expected"),
    [(torch.ones(1, HIDDEN + 1), 0), (torch.ones(1, 2, HIDDEN), -1)],
    ids=["wrong-width", "position-specific"],
)
def test_a_vector_outside_the_table(activations: torch.Tensor, expected: int):
    """A bad vector gets row 0 (no steering); a position-specific one gets -1."""
    table = _SteerTable(device=torch.device("cpu"), hidden=HIDDEN)
    vector = SteeringVector(activations=activations, layer_indices=[LAYER])
    assert table.row(b"a", [vector]) == expected
    # A failed row goes back to the free list; no row stays taken.
    assert table.row(b"b", [_vector(0)]) == 1


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
    rows = [table.row(bytes([n]), [config]) for n, config in enumerate(configs)]
    batched = output.clone()
    layout = _token_layout(spans, rows, torch.device("cpu"))
    _apply_table_rows(table, LAYER, layout, batched, norm_ref)
    torch.testing.assert_close(batched, expected, atol=1e-4, rtol=1e-4)
    assert torch.equal(batched[4:6], output[4:6])

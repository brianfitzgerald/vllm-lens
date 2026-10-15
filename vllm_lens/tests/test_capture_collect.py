"""Capture collection needs no GPU: rows go to the right request, and the batch RPC finds
each request's capture by its external id."""

import pickle
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_lens._helpers._serialize import unpack_payloads
from vllm_lens._worker_ext import HiddenStatesExtension, _capture_rows

HIDDEN = 4


def _extension(extras: dict[str, dict[str, Any] | None]) -> SimpleNamespace:
    """A stand-in extension whose runner holds one request per entry of ``extras``."""
    requests = {
        req_id: SimpleNamespace(
            sampling_params=SimpleNamespace(extra_args=extra),
            num_computed_tokens=0,
            num_prompt_tokens=3,
        )
        for req_id, extra in extras.items()
    }
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(num_reqs=len(extras), req_ids=list(extras)),
        requests=requests,
    )
    return SimpleNamespace(
        model_runner=runner,
        _captured_states={},
        _pooled_counts={},
        _projections=set(),
        _direction_tables={"d": torch.eye(HIDDEN)[:2]},
    )


def test_rows_land_on_their_requests() -> None:
    """Full, pooled, projected and uncaptured requests in one step each get their own rows."""
    extension = _extension(
        {
            "a-1": {"output_residual_stream": [5]},
            "b-1": None,
            "c-1": {"output_residual_stream": "[5]"},
            "d-1": {
                "output_residual_stream": [5],
                "output_residual_stream_pool": "last",
            },
            "e-1": {"output_residual_stream": [5], "_projection_id": "d"},
            "f-1": {"output_residual_stream": [6]},
        }
    )
    hidden = torch.arange(18 * HIDDEN, dtype=torch.float32).reshape(18, HIDDEN)
    bounds = torch.tensor([0, 3, 6, 9, 12, 15, 18])
    _capture_rows(extension, 5, hidden, bounds)  # type: ignore[arg-type]

    states = extension._captured_states
    assert set(states) == {"a-1", "c-1", "d-1", "e-1"}
    assert torch.equal(states["a-1"][5][0], hidden[0:3])
    assert torch.equal(states["c-1"][5][0], hidden[6:9])
    assert torch.equal(states["d-1"][5][0], hidden[11:12])
    projected = states["e-1"][5][0]
    assert torch.equal(projected[:, :2], hidden[12:15, :2])
    assert torch.allclose(projected[:, 2], hidden[12:15].norm(dim=-1))


@pytest.mark.parametrize(
    ("internal", "asked", "found"),
    [
        (["0-ab", "1-cd", "2-ef"], ["0", "2"], {"0": "0-ab", "2": "2-ef"}),
        (["cmpl-x-ab", "cmpl-y-cd"], ["cmpl-x"], {"cmpl-x": "cmpl-x-ab"}),
        (["0-ab"], ["9"], {}),
        (["10-ab", "1-cd"], ["1"], {"1": "1-cd"}),
    ],
    ids=["offline", "dashed-external", "miss", "prefix-is-not-a-match"],
)
def test_batch_finds_each_external_id(
    internal: list[str], asked: list[str], found: dict[str, str]
) -> None:
    """Each asked external id gets the capture of its internal id, and no other; a capture's
    per-step rows come back as one (layers, positions, hidden) tensor."""
    rows = {req_id: float(n) for n, req_id in enumerate(internal)}
    extension = SimpleNamespace(
        _captured_states={
            req_id: {
                2: [torch.full((2, HIDDEN), value), torch.full((1, HIDDEN), value)],
                7: [torch.full((3, HIDDEN), -value)],
            }
            for req_id, value in rows.items()
        },
        _pooled_counts={},
        _projections=set(),
    )
    result = HiddenStatesExtension.get_captured_states_batch(extension, asked)  # type: ignore[arg-type]
    payloads = unpack_payloads(pickle.loads(result)) if result else {}
    assert set(payloads) == set(found)
    for external, req_id in found.items():
        stream = payloads[external]["activations"]["residual_stream"]
        assert stream.shape == (2, 3, HIDDEN)
        assert torch.equal(stream[0], torch.full((3, HIDDEN), rows[req_id]))
        assert torch.equal(stream[1], torch.full((3, HIDDEN), -rows[req_id]))


def test_batch_returns_pooled_means_in_layer_order() -> None:
    """A pooled capture comes back as one mean row per layer, in layer order and the capture
    dtype, beside a full capture of another request."""
    sums = {7: torch.full((1, HIDDEN), 6.0), 2: torch.full((1, HIDDEN), 4.0)}
    extension = SimpleNamespace(
        _captured_states={
            "0-ab": {layer: [total] for layer, total in sums.items()},
            "1-cd": {2: [torch.ones(3, HIDDEN, dtype=torch.bfloat16)]},
        },
        _pooled_counts={"0-ab": {7: (3, torch.bfloat16, 3), 2: (2, torch.bfloat16, 2)}},
        _projections=set(),
    )
    result = HiddenStatesExtension.get_captured_states_batch(extension, ["0", "1"])  # type: ignore[arg-type]
    payloads = unpack_payloads(pickle.loads(result))
    pooled = payloads["0"]["activations"]["residual_stream"]
    assert pooled.dtype == torch.bfloat16 and pooled.shape == (2, 1, HIDDEN)
    assert torch.equal(pooled[:, 0, 0], torch.tensor([2.0, 2.0], dtype=torch.bfloat16))
    assert payloads["1"]["activations"]["residual_stream"].shape == (1, 3, HIDDEN)
    assert extension._pooled_counts == {} and extension._captured_states == {}

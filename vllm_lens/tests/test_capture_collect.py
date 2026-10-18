"""Capture collection needs no GPU: rows go to the right request, and the batch RPC finds
each request's capture by its external id."""

import os
from typing import Any

import pytest
import torch

from vllm_lens._helpers._transport import Segment, open_segment, unpack_payloads
from vllm_lens._worker_ext import HiddenStatesExtension, _capture_rows

from .conftest import fake_extension

HIDDEN = 4


def _collect(extension: Any, asked: list[str]) -> dict[str, Any]:
    """Run the collect RPC and map its memfd as the offline plugin does, with one rank."""
    result = HiddenStatesExtension.get_captured_states_batch(extension, asked)  # type: ignore[arg-type]
    if result is None:
        return {}
    segment, packed = result
    region = open_segment(Segment(*segment)) if segment else None
    if segment:
        os.close(extension._segment_fd)
    return unpack_payloads(packed, region)


def test_rows_land_on_their_requests() -> None:
    """Full, pooled, projected and uncaptured requests in one step each get their own rows."""
    extension = fake_extension(
        {
            "a-1": {"output_residual_stream": [5]},
            "b-1": None,
            "c-1": {"output_residual_stream": "[5]"},
            "d-1": {
                "output_residual_stream": [5],
                "output_residual_stream_pool": "last",
            },
            "e-1": {
                "output_residual_stream": [5],
                "output_residual_stream_project": "d",
            },
            "f-1": {"output_residual_stream": [6]},
        },
        n_prompt=3,
        directions={"d": torch.eye(HIDDEN)[:2]},
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
    extension = fake_extension({})
    extension._captured_states = {
        req_id: {
            2: [torch.full((2, HIDDEN), value), torch.full((1, HIDDEN), value)],
            7: [torch.full((3, HIDDEN), -value)],
        }
        for req_id, value in rows.items()
    }
    payloads = _collect(extension, asked)
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
    extension = fake_extension({})
    extension._captured_states = {
        "0-ab": {layer: [total] for layer, total in sums.items()},
        "1-cd": {2: [torch.ones(3, HIDDEN, dtype=torch.bfloat16)]},
    }
    extension._pooled = {"0-ab": (torch.bfloat16, 2)}
    payloads = _collect(extension, ["0", "1"])
    pooled = payloads["0"]["activations"]["residual_stream"]
    assert pooled.dtype == torch.bfloat16 and pooled.shape == (2, 1, HIDDEN)
    assert torch.equal(pooled[:, 0, 0], torch.tensor([2.0, 3.0], dtype=torch.bfloat16))
    assert payloads["1"]["activations"]["residual_stream"].shape == (1, 3, HIDDEN)
    assert extension._pooled == {} and extension._captured_states == {}


@pytest.mark.parametrize("external_ids", [["0"], None], ids=["one", "all"])
def test_clear_request_state_drops_pooled_and_projected_state(
    external_ids: list[str] | None,
) -> None:
    """Clearing a request drops its pooled divisor and projection mark; None clears all."""
    worker = HiddenStatesExtension.__new__(HiddenStatesExtension)
    worker._captured_qk, worker._hook_contexts = {}, {}
    worker._segment_fd = None
    worker._captured_states = {"0-ab": {2: []}, "1-cd": {2: []}}
    worker._pooled = {"0-ab": (torch.float32, 1), "1-cd": (torch.float32, 1)}
    worker._projections = {"0-ab", "1-cd"}
    worker.clear_request_state(external_ids)
    kept = set() if external_ids is None else {"1-cd"}
    assert set(worker._captured_states) == kept
    assert set(worker._pooled) == kept
    assert worker._projections == kept

"""The rows a capture stores for each forward step. No GPU."""

import pytest
import torch

from vllm_lens._worker_ext import HiddenStatesExtension, _capture_rows

from .conftest import LAYER_IDX, fake_extension

# Forward steps as (first absolute position, number of rows). The prompt has
# 10 positions; positions 10 and 11 are decode steps.
STEP_SCHEDULES = {
    "single-pass": [(0, 10), (10, 1), (11, 1)],
    "chunked": [(0, 4), (4, 4), (8, 2), (10, 1)],
    # A preemption after two chunks: the prefill starts again at position 0.
    "preempted": [(0, 4), (4, 4), (0, 4), (4, 4), (8, 2), (10, 1)],
    # The last chunk also holds the first decode position.
    "chunk-past-the-prompt": [(0, 6), (6, 5)],
}


def _step(extension, rows: torch.Tensor, first: int) -> None:
    """Capture ``rows`` as one forward step of ``r-0`` that starts at position ``first``."""
    extension.model_runner.requests["r-0"].num_computed_tokens = first
    _capture_rows(extension, LAYER_IDX, rows, torch.tensor([0, rows.shape[0]]))


@pytest.mark.parametrize("schedule", STEP_SCHEDULES)
@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        (
            {},
            lambda s, steps: torch.cat([s[first : first + n] for first, n in steps]),
        ),
    ],
    ids=["full"],
)
def test_each_position_is_captured_once(schedule, extra, expected):
    """The capture is exact for any step schedule, with a re-prefill too; a full capture
    keeps the rows of every step."""
    stream = torch.randn(12, 8, generator=torch.Generator().manual_seed(0)).to(
        torch.bfloat16
    )
    extension = fake_extension(
        {"r-0": {"output_residual_stream": [LAYER_IDX], **extra}}
    )
    for first, n_rows in STEP_SCHEDULES[schedule]:
        _step(extension, stream[first : first + n_rows], first)

    activations = HiddenStatesExtension._build_payload(extension, "r-0")["activations"]  # type: ignore[arg-type]
    got = activations["residual_stream"][0]
    want = expected(stream.float(), STEP_SCHEDULES[schedule])
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got.float(), want, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    ("layers", "stored"), [([LAYER_IDX], True), ([LAYER_IDX + 1], False), (True, True)]
)
def test_only_captured_layers_are_stored(layers, stored) -> None:
    """A layer is stored for a request that lists it or captures every layer."""
    extension = fake_extension({"r-0": {"output_residual_stream": layers}, "r-1": None})
    extension.model_runner.input_batch.num_reqs = 2
    _capture_rows(extension, LAYER_IDX, torch.ones(3, 8), torch.tensor([0, 2, 3]))  # type: ignore[arg-type]
    assert ("r-0" in extension._captured_states) is stored
    assert "r-1" not in extension._captured_states

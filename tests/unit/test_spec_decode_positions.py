"""CPU tests for the rejection of absolute positions under async speculative decoding."""

from types import SimpleNamespace

import pytest
import torch

import vllm_lens._activations_plugin as plugin
from vllm_lens._helpers.types import SteeringVector

PLAIN = SteeringVector(activations=torch.zeros(1, 4), layer_indices=[0])
INDEXED = SteeringVector(
    activations=torch.zeros(1, 1, 4), layer_indices=[0], position_indices=[5]
)
SEQUENTIAL = SteeringVector(activations=torch.zeros(1, 2, 4), layer_indices=[0])


@pytest.mark.parametrize(
    ("speculative", "async_scheduling", "vectors", "positions", "rejected"),
    [
        (True, True, [INDEXED], None, True),
        (True, True, [PLAIN, SEQUENTIAL], None, True),
        (True, True, [PLAIN], None, False),
        (True, True, None, None, False),
        (True, True, None, [3, 9], True),
        (True, True, None, "last_prompt_token", False),
        (True, False, [INDEXED], [3], False),
        (False, True, [INDEXED], [3], False),
    ],
    ids=[
        "indexed",
        "sequential",
        "plain",
        "none",
        "capture-list",
        "capture-last-prompt",
        "sync",
        "no-spec",
    ],
)
def test_positions_under_async_spec_decode(
    speculative, async_scheduling, vectors, positions, rejected
):
    config = SimpleNamespace(
        speculative_config=object() if speculative else None,
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
    )
    if rejected:
        with pytest.raises(ValueError, match="async speculative decoding"):
            plugin._reject_async_spec_positions(config, vectors, positions)
    else:
        plugin._reject_async_spec_positions(config, vectors, positions)

"""CPU tests for the rejection of position-specific steering under async speculative decoding."""

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
    ("speculative", "async_scheduling", "vectors", "rejected"),
    [
        (True, True, [INDEXED], True),
        (True, True, [PLAIN, SEQUENTIAL], True),
        (True, True, [PLAIN], False),
        (True, True, None, False),
        (True, False, [INDEXED], False),
        (False, True, [INDEXED], False),
    ],
    ids=["indexed", "sequential", "plain", "none", "sync", "no-spec"],
)
def test_position_vectors_under_async_spec_decode(
    speculative, async_scheduling, vectors, rejected
):
    config = SimpleNamespace(
        speculative_config=object() if speculative else None,
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
    )
    if rejected:
        with pytest.raises(ValueError, match="async speculative decoding"):
            plugin._reject_async_spec_positions(config, vectors)
    else:
        plugin._reject_async_spec_positions(config, vectors)

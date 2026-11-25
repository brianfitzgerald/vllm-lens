"""CPU tests for the rejection of positions and capture under async speculative decoding."""

from types import SimpleNamespace

import pytest
import torch

import vllm_lens._activations_plugin as plugin
from vllm_lens._helpers.types import SteeringVector

PLAIN = SteeringVector(activations=torch.zeros(1, 4), layer_indices=[0])
INDEXED = SteeringVector(
    activations=torch.zeros(1, 4), layer_indices=[0], position_indices=[5]
)


@pytest.mark.parametrize(
    ("speculative", "async_scheduling", "vectors", "capture", "rejected"),
    [
        (True, True, [INDEXED], None, True),
        (True, True, [PLAIN], None, False),
        (True, True, None, None, False),
        (True, True, None, {}, True),
        (True, False, [INDEXED], {}, False),
        (False, True, [INDEXED], {}, False),
    ],
    ids=[
        "indexed",
        "plain",
        "none",
        "capture",
        "sync",
        "no-spec",
    ],
)
def test_positions_under_async_spec_decode(
    speculative, async_scheduling, vectors, capture, rejected
):
    config = SimpleNamespace(
        speculative_config=object() if speculative else None,
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
    )
    extra = {} if capture is None else {"output_residual_stream": [0], **capture}
    if rejected:
        with pytest.raises(ValueError, match="async speculative decoding"):
            plugin._reject_async_spec_positions(config, vectors, extra)
    else:
        plugin._reject_async_spec_positions(config, vectors, extra)

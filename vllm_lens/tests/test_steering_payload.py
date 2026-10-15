"""A steering vector's payload carries that vector alone. No GPU."""

import pickle

import torch

from vllm_lens import SteeringVector
from vllm_lens._activations_plugin import _compact_vectors


def test_a_row_view_pickles_as_the_row_alone():
    """A row of a large matrix is copied out, so its pickle is not the whole matrix."""
    matrix = torch.randn(100, 64)
    view = SteeringVector(
        activations=matrix[3].unsqueeze(0), layer_indices=[0], scale=2.0
    )
    whole = SteeringVector(activations=torch.randn(1, 64), layer_indices=[0])
    compact = _compact_vectors([view, whole])
    assert len(pickle.dumps(compact[0])) < len(pickle.dumps(view)) / 10
    torch.testing.assert_close(compact[0].activations, view.activations)
    assert compact[0].scale == 2.0 and compact[0].layer_indices == [0]
    assert compact[1] is whole

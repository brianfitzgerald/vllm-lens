"""Norm matching uses the unsteered stream for all vectors in a layer."""

import pickle
from types import SimpleNamespace

import pytest
import torch

from vllm_lens import SteeringVector


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_vectors_use_unsteered_norm(worker, monkeypatch, dtype, fused, reverse):
    """Mixed vectors and duplicate positions use the same original stream norm."""
    hidden = torch.tensor([[3.0, 4.0], [0.0, 2.0]], dtype=dtype)
    original = hidden.clone()
    residual = torch.ones_like(hidden) if fused else None
    output = (hidden, residual) if fused else hidden
    ext = worker.HiddenStatesExtension()
    ext.model_runner = SimpleNamespace(
        model=torch.nn.Linear(2, 2, bias=False).to(dtype),
        model_config=SimpleNamespace(get_hidden_size=lambda: 2),
        input_batch=SimpleNamespace(num_reqs=1, req_ids=["request-0"]),
        requests={
            "request-0": SimpleNamespace(
                sampling_params=SimpleNamespace(extra_args={"_steering_id": "a"}),
                num_computed_tokens=3,
            )
        },
        _model_forward=lambda: worker._hook_inner(ext, 2, output),
    )
    ext.model_config = SimpleNamespace(get_total_num_hidden_layers=lambda: 4)
    ext.parallel_config = SimpleNamespace(tensor_parallel_size=1)
    ext.rank = 0
    monkeypatch.setattr(worker, "_discover_layer_modules", lambda _: {})
    monkeypatch.setattr(worker, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        worker,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={
                "layer": SimpleNamespace(query_start_loc=torch.tensor([0, 2]))
            }
        ),
    )
    ext.install_hooks()
    configs = [
        SteeringVector(activations=torch.tensor([[1.0, 0.0]]), layer_indices=[2]),
        SteeringVector(
            activations=torch.tensor([[0.0, 1.0]]),
            layer_indices=[2],
            norm_match=True,
            scale=0.5,
        ),
        SteeringVector(
            activations=torch.eye(2).unsqueeze(0),
            layer_indices=[2],
            position_indices=[3, 3],
            norm_match=True,
            scale=0.25,
        ),
    ]
    ext.set_steering_data("a", pickle.dumps(configs[::-1] if reverse else configs))
    result = ext.model_runner._model_forward()
    actual = result[0] if fused else result
    stream = original if residual is None else original + residual
    directions = torch.tensor([[0.25, 0.75], [0.0, 0.5]])
    expected = original + (
        torch.tensor([1.0, 0.0])
        + directions * stream.float().norm(dim=-1, keepdim=True)
    ).to(dtype)
    torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.01)
    assert torch.equal(hidden, original)
    if fused:
        assert torch.equal(result[1], residual)

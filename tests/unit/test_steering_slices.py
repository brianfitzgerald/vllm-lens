"""Steering slices and request cleanup without vLLM or CUDA."""

import pickle
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_lens import SteeringVector


def extension(worker, dtype=torch.float32):
    """Create a CPU worker with a small model and empty request data."""
    return SimpleNamespace(
        model_runner=SimpleNamespace(
            model=torch.nn.Linear(8, 8, bias=False).to(dtype),
            model_config=SimpleNamespace(get_hidden_size=lambda: 8),
        ),
        model_config=SimpleNamespace(get_total_num_hidden_layers=lambda: 4),
        _steering_data={},
        _steering_tables={},
        _row_steering={},
    )


def vector(seed, positional=False, matched=False, scale=1.0):
    """Create a deterministic flat or positional vector on layer 2."""
    values = torch.randn(
        (1, 6, 8) if positional else (1, 8),
        generator=torch.Generator().manual_seed(seed),
    )
    return SteeringVector(
        activations=values,
        layer_indices=[2],
        scale=scale,
        norm_match=matched,
        position_indices=[1, 4, 5, 9, 2, 3] if positional else None,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("kind", ["add", "norm", "mixed", "positional", "combined"])
def test_slices_match_independent_sum(worker, dtype, fused, kind):
    """Slices equal the independent per-token sum, including overlapping positions."""
    configs = {
        "add": [vector(0, scale=-2)],
        "norm": [vector(1, matched=True, scale=0.5)],
        "mixed": [vector(0), vector(1, matched=True, scale=-2)],
        "positional": [
            vector(2, positional=True),
            vector(3, positional=True, matched=True),
        ],
        "combined": [
            vector(0),
            vector(1, matched=True),
            vector(2, positional=True),
            vector(3, positional=True, matched=True),
        ],
    }[kind]
    ext = extension(worker, dtype)
    worker.HiddenStatesExtension.set_steering_data(ext, "a", pickle.dumps(configs))
    hidden = torch.randn(5, 8, generator=torch.Generator().manual_seed(10)).to(dtype)
    residual = (
        torch.randn(5, 8, generator=torch.Generator().manual_seed(11)).to(dtype)
        if fused
        else None
    )
    original = hidden.clone()
    stream = original if residual is None else original + residual
    additions = torch.zeros_like(hidden, dtype=torch.float32)
    directions = torch.zeros_like(additions)
    for cfg in configs:
        values = cfg.activations[0].to(dtype).float()
        for token in range(3):
            absolute = 3 + token
            selected = (
                [values]
                if values.dim() == 1
                else [
                    values[i]
                    for i, pos in enumerate(cfg.position_indices)
                    if pos == absolute
                ]
            )
            for value in selected:
                if cfg.norm_match:
                    directions[token + 1] += value / (value.norm() + 1e-6) * cfg.scale
                else:
                    additions[token + 1] += value * cfg.scale
    expected = original + (
        additions + directions * stream.float().norm(dim=-1, keepdim=True)
    ).to(dtype)
    add, norm = worker._steering_buffers(
        [worker._SteerSlice(1, 4, 3, ext._steering_data["a"][2])], 5
    )
    worker._steer(hidden, residual, add, norm)
    torch.testing.assert_close(
        hidden,
        expected,
        atol=1e-6 if dtype == torch.float32 else 0.02,
        rtol=1e-5 if dtype == torch.float32 else 0.02,
    )
    assert torch.equal(hidden[[0, 4]], original[[0, 4]])


@pytest.mark.parametrize("offline", [False, True])
def test_plan_tracks_reordered_batches_and_cleanup(worker, monkeypatch, offline):
    """Reordering and cleanup cannot reuse another request's steering."""
    ext = extension(worker)
    worker.HiddenStatesExtension.set_steering_data(ext, "a", pickle.dumps([vector(0)]))
    ext.model_runner.requests = {
        f"{key}-0": SimpleNamespace(
            sampling_params=SimpleNamespace(
                extra_args={"_steering_id": key} if offline else {}
            ),
            num_computed_tokens=done,
        )
        for key, done in [("a", 3), ("b", 0)]
    }
    ext.model_runner.input_batch = SimpleNamespace(num_reqs=2, req_ids=["a-0", "b-0"])
    monkeypatch.setattr(
        worker, "_batch_layout", lambda *a, **k: torch.tensor([0, 3, 5])
    )
    plan = worker._steer_pass(ext)
    assert [(s.start, s.end, s.abs_start) for s in plan[2]] == [(0, 3, 3)]
    ext.model_runner.input_batch.req_ids.reverse()
    plan = worker._steer_pass(ext)
    assert [(s.start, s.end, s.abs_start) for s in plan[2]] == [(3, 5, 3)]
    worker.HiddenStatesExtension.clear_steering_data(ext, "a")
    assert worker._steer_pass(ext) is None
    worker.HiddenStatesExtension.set_steering_data(ext, "a", pickle.dumps([vector(1)]))
    torch.testing.assert_close(ext._steering_data["a"][2].add, vector(1).activations)


def test_all_positions_are_applied_without_capacity_limit(worker):
    """More than 512 positions and duplicate positions all contribute."""
    ext = extension(worker)
    cfg = SteeringVector(
        activations=torch.ones(1, 1025, 8),
        layer_indices=[2],
        position_indices=list(range(1024)) + [1023],
    )
    worker.HiddenStatesExtension.set_steering_data(ext, "a", pickle.dumps([cfg]))
    add, norm = worker._steering_buffers(
        [worker._SteerSlice(0, 1024, 0, ext._steering_data["a"][2])], 1024
    )
    assert norm is None
    assert torch.equal(add[:1023], torch.ones(1023, 8))
    assert torch.equal(add[1023], torch.full((8,), 2.0))


@pytest.mark.parametrize(
    "width,layer,message", [(9, 2, "vector width"), (8, 4, "out of range")]
)
def test_invalid_replacement_preserves_request(worker, width, layer, message):
    """Invalid vectors leave the previous request data intact."""
    ext = extension(worker)
    worker.HiddenStatesExtension.set_steering_data(ext, "a", pickle.dumps([vector(0)]))
    original = ext._steering_data["a"]
    bad = SteeringVector(activations=torch.ones(1, width), layer_indices=[layer])
    with pytest.raises(ValueError, match=message):
        worker.HiddenStatesExtension.set_steering_data(ext, "a", pickle.dumps([bad]))
    assert ext._steering_data["a"] is original


def test_flat_addition_does_not_compute_a_norm(worker, monkeypatch):
    """Flat addition does not launch norm operations during a forward pass."""
    ext = extension(worker)
    worker.HiddenStatesExtension.set_steering_data(ext, "a", pickle.dumps([vector(0)]))
    add, norm = worker._steering_buffers(
        [worker._SteerSlice(0, 3, 0, ext._steering_data["a"][2])], 3
    )
    monkeypatch.setattr(
        torch.linalg, "vector_norm", lambda *a, **k: pytest.fail("unexpected norm")
    )
    hidden = torch.zeros(3, 8)
    worker._steer(hidden, None, add, norm)
    torch.testing.assert_close(hidden, vector(0).activations[0].expand(3, -1))


@pytest.mark.parametrize("matched", [False, True])
def test_batch_buffers_clear_gaps_and_padding(worker, matched):
    """Batch writes preserve buffer addresses and clear gaps and padding."""
    ext = extension(worker)
    worker.HiddenStatesExtension.set_steering_data(
        ext, "a", pickle.dumps([vector(0, matched=matched)])
    )
    data = ext._steering_data["a"][2]
    spans = [worker._SteerSlice(1, 3, 0, data), worker._SteerSlice(5, 6, 0, data)]
    add = torch.full((8, 8), 99.0)
    norm = torch.full_like(add, 99.0)
    pointers = (add.data_ptr(), norm.data_ptr())
    worker._steering_buffers(spans, 8, add, norm)
    assert pointers == (add.data_ptr(), norm.data_ptr())
    assert torch.count_nonzero(add[[0, 3, 4, 6, 7]]) == 0
    assert torch.count_nonzero(norm[[0, 3, 4, 6, 7]]) == 0
    expected_add, expected_norm = data.add, data.norm
    torch.testing.assert_close(add[1:3], expected_add.expand(2, -1))
    if matched:
        torch.testing.assert_close(norm[1:3], expected_norm.expand(2, -1))
    else:
        assert torch.count_nonzero(norm) == 0
    worker._steering_buffers([worker._SteerSlice(0, 1, 0, data)], 8, add, norm)
    assert torch.count_nonzero(add[1:]) == 0
    assert torch.count_nonzero(norm[1:]) == 0


def test_forward_plan_is_cleared_after_model_failure(worker, monkeypatch):
    """A failed model call cannot leave steering for the next forward pass."""
    ext = worker.HiddenStatesExtension()
    ext.model_runner = extension(worker).model_runner
    ext.model_runner.input_batch = SimpleNamespace(num_reqs=0, req_ids=[])
    original = Mock(side_effect=["result", RuntimeError("model failed")])
    ext.model_runner._model_forward = original
    ext.parallel_config = SimpleNamespace(tensor_parallel_size=1)
    ext.rank = 0
    plan = {2: []}
    monkeypatch.setattr(worker, "_discover_layer_modules", lambda _: {})
    monkeypatch.setattr(worker, "_steer_pass", lambda _: plan)
    ext.install_hooks()
    assert ext.model_runner._model_forward() == "result"
    assert ext._steer_pass is None
    with pytest.raises(RuntimeError, match="model failed"):
        ext.model_runner._model_forward()
    assert ext._steer_pass is None
    assert original.call_count == 2

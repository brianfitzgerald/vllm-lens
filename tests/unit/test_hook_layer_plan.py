"""Hook plans contain only the layers that run hooks."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("pre", [False, True])
@pytest.mark.parametrize("persistent", [False, True])
def test_hook_layer_plan(worker, monkeypatch, pre, persistent):
    """An unrelated layer returns before reading attention metadata."""
    hook = SimpleNamespace(pre=pre, layer_indices=[3])
    ext = SimpleNamespace(
        model_runner=SimpleNamespace(
            input_batch=SimpleNamespace(num_reqs=1, req_ids=["r-0"]),
            requests={
                "r-0": SimpleNamespace(
                    sampling_params=SimpleNamespace(extra_args={"_hook_id": "h"})
                )
            },
        ),
        _persistent_hooks=[hook] if persistent else [],
        _hook_data={} if persistent else {"h": [hook]},
        _steer_pass=None,
        _should_capture=True,
    )
    ext._hook_pass = worker._hook_pass(ext)
    assert (ext._hook_pass.pre if pre else ext._hook_pass.post) == frozenset({3})
    monkeypatch.setattr(
        worker, "_query_starts", lambda *a, **k: pytest.fail("idle layer")
    )
    assert worker._hook_inner(ext, 2, torch.zeros(1, 2)) is None
    assert worker._pre_hook_inner(ext, 2, torch.zeros(1, 2)) is None


def test_hook_plan_failure_keeps_steering(worker, monkeypatch):
    """A failed hook plan leaves the step's steering plan in place."""
    seen = []
    ext = worker.HiddenStatesExtension()
    ext.model_runner = SimpleNamespace(
        _model_forward=lambda: seen.append((ext._steer_pass, ext._hook_pass))
    )
    ext.parallel_config = SimpleNamespace(tensor_parallel_size=1)
    ext.rank = 0
    ext._init_state()
    plan = object()
    monkeypatch.setattr(worker, "_steer_pass", lambda _ext: plan)

    def fail(_ext):
        raise RuntimeError("bad hook plan")

    monkeypatch.setattr(worker, "_hook_pass", fail)
    ext.model_runner._model_forward()
    assert seen == [(plan, None)]
    assert ext._steer_pass is None

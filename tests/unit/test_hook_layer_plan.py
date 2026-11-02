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
        worker, "_batch_layout", lambda *a, **k: pytest.fail("idle layer")
    )
    assert worker._hook_inner(ext, 2, torch.zeros(1, 2)) is None
    assert worker._pre_hook_inner(ext, 2, torch.zeros(1, 2)) is None

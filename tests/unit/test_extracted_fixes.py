"""Steering cleanup and parallel capture validation."""

from types import SimpleNamespace

import pytest
import torch

import vllm_lens._activations_plugin as plugin
from vllm_lens import SteeringVector


@pytest.mark.parametrize("method", ["generate", "chat"])
@pytest.mark.parametrize("stage", ["generation", "output"])
def test_vector_cleanup(monkeypatch, stub_engine, method, stage):
    """Inline steering data is cleared when generation or the output step fails."""
    calls = []
    engine = stub_engine(
        lambda name, args=(): calls.append((name, args)) or [], offline=True
    )
    params = SimpleNamespace(
        n=1,
        extra_args={
            "apply_steering_vectors": [
                SteeringVector(activations=torch.ones(1, 2), layer_indices=[0])
            ]
        },
        skip_reading_prefix_cache=False,
    )

    def fail(*args, **kwargs):
        raise RuntimeError("failed")

    monkeypatch.setattr(
        plugin,
        "_original_llm_" + method,
        fail if stage == "generation" else lambda *a, **k: [],
    )
    if stage == "output":
        monkeypatch.setattr(plugin, "_finalize_offline_outputs", fail)
    with pytest.raises(RuntimeError, match="failed"):
        getattr(plugin, "_patched_llm_" + method)(engine, ["p"], params)
    sent = [args for name, args in calls if name == "set_steering_data"]
    assert [("clear_steering_data", (key,)) for key, _ in sent] == [
        call for call in calls if call[0] == "clear_steering_data"
    ]
    assert len(sent) == 1
    assert "apply_steering_vectors" in params.extra_args


def test_parallel_capture_rejected():
    """Capture with n > 1 is refused."""
    with pytest.raises(ValueError, match="n > 1"):
        plugin._reject_parallel_capture(
            SimpleNamespace(n=2), {"output_residual_stream": [0]}
        )

"""Steering cleanup and parallel capture validation."""

from types import SimpleNamespace
import pytest
import torch
from vllm_lens import SteeringVector
import vllm_lens._activations_plugin as plugin


@pytest.mark.parametrize("method", ["generate", "chat"])
@pytest.mark.parametrize("stage", ["generation", "output"])
def test_vector_cleanup(monkeypatch, method, stage):
    calls = []
    engine = SimpleNamespace(
        _hooks_installed=True,
        llm_engine=SimpleNamespace(vllm_config=SimpleNamespace(additional_config=None)),
        collective_rpc=lambda name, args=(): calls.append((name, args)) or [],
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
    assert ("clear_steering_data", ("_steer_0",)) in calls
    assert "apply_steering_vectors" in params.extra_args


@pytest.mark.parametrize(
    "extra,hooks",
    [
        ({"output_residual_stream": [0]}, None),
        ({"output_qk": True}, None),
        ({}, ["hook"]),
    ],
)
def test_parallel_capture_rejected(extra, hooks):
    with pytest.raises(ValueError, match="n > 1"):
        plugin._reject_parallel_capture(SimpleNamespace(n=2), extra, hooks)

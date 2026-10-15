"""Engines with different armed layers get different compile cache keys: the settings are in
``additional_config``, which ``VllmConfig.compute_hash`` covers. No engine starts."""

import pytest
from vllm import EngineArgs

from .conftest import MODEL_NAME


def _config_hash(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> str:
    """The compile cache key of an engine config built with ``env``."""
    for name in ("VLLM_LENS_CAPTURE_LAYERS", "VLLM_LENS_STEER_LAYERS"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return EngineArgs(model=MODEL_NAME).create_engine_config().compute_hash()


def test_each_layer_set_gets_its_own_compile_key(monkeypatch: pytest.MonkeyPatch):
    """Two steer sets and a capture set give three keys; the same set gives the same key."""
    envs = [
        {"VLLM_LENS_STEER_LAYERS": "2"},
        {"VLLM_LENS_STEER_LAYERS": "5"},
        {"VLLM_LENS_CAPTURE_LAYERS": "2"},
    ]
    keys = [_config_hash(monkeypatch, env) for env in envs]
    assert len(set(keys)) == 3
    assert _config_hash(monkeypatch, envs[0]) == keys[0]

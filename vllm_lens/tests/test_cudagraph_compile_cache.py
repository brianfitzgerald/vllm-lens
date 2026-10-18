"""Engines with different armed layers get different compile cache keys: the settings are in
``additional_config``, which ``VllmConfig.compute_hash`` covers. No engine starts."""

from vllm import EngineArgs

from .conftest import MODEL_NAME


def _config_hash(lens: dict[str, list[int]]) -> str:
    """The compile cache key of an engine config with ``lens`` as its lens settings."""
    args = EngineArgs(model=MODEL_NAME, additional_config={"vllm_lens": lens})
    return args.create_engine_config().compute_hash()


def test_each_layer_set_gets_its_own_compile_key():
    """Two steer sets and a capture set give three keys; the same set gives the same key."""
    sets = [{"steer_layers": [2]}, {"steer_layers": [5]}, {"capture_layers": [2]}]
    keys = [_config_hash(lens) for lens in sets]
    assert len(set(keys)) == 3
    assert _config_hash(sets[0]) == keys[0]

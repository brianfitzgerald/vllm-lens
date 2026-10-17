"""CPU tests for ``_patched_create_engine_config`` environment defaults."""

from types import SimpleNamespace

import pytest
import torch

import vllm_lens._activations_plugin as plugin
from vllm_lens import SteeringVector
from vllm_lens._cudagraph import CONFIG_KEY, LensGraphConfig


def _fake_config(dtype):
    return SimpleNamespace(
        model_config=SimpleNamespace(dtype=dtype), use_v2_model_runner=False
    )


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(
        plugin, "_original_create_engine_config", lambda self, *a, **k: self._cfg
    )
    monkeypatch.delenv("TRITON_F32_DEFAULT", raising=False)
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)

    def run(dtype):
        args = SimpleNamespace(
            worker_extension_cls=None,
            enforce_eager=False,
            additional_config=None,
            _cfg=_fake_config(dtype),
        )
        plugin._patched_create_engine_config(args)
        return args

    return run


def test_fp32_model_defaults_triton_to_ieee(patched, monkeypatch):
    import os

    patched(torch.float32)
    assert os.environ["TRITON_F32_DEFAULT"] == "ieee"


def test_bf16_model_leaves_triton_alone(patched):
    import os

    patched(torch.bfloat16)
    assert "TRITON_F32_DEFAULT" not in os.environ


def test_explicit_triton_setting_wins(patched, monkeypatch):
    import os

    monkeypatch.setenv("TRITON_F32_DEFAULT", "tf32")
    patched(torch.float32)
    assert os.environ["TRITON_F32_DEFAULT"] == "tf32"


def test_still_forces_eager_and_worker_ext(patched):
    args = patched(torch.bfloat16)
    assert args.enforce_eager is True
    assert args.worker_extension_cls == plugin._WORKER_EXT


@pytest.mark.parametrize(
    ("env", "enforce_eager"),
    [
        ({}, True),
        ({"VLLM_LENS_CAPTURE_LAYERS": "2"}, False),
        ({"VLLM_LENS_STEER_LAYERS": "3"}, False),
    ],
    ids=["default", "capture-layers", "steer-layers"],
)
def test_a_layer_list_keeps_compilation_on(patched, monkeypatch, env, enforce_eager):
    """A layer list leaves enforce_eager off and stores the settings for the workers."""
    for name in ("VLLM_LENS_CAPTURE_LAYERS", "VLLM_LENS_STEER_LAYERS"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    args = patched(torch.bfloat16)
    assert args.enforce_eager is enforce_eager
    assert (CONFIG_KEY in (args.additional_config or {})) is not enforce_eager
    assert LensGraphConfig.from_vllm_config(args) == LensGraphConfig.from_env()


@pytest.mark.parametrize(
    ("residual_stream", "steering", "match"),
    [
        ("[2]", None, None),
        ("[5]", None, "VLLM_LENS_CAPTURE_LAYERS"),
        ("true", None, "must be a list"),
        (None, [4], "VLLM_LENS_STEER_LAYERS"),
    ],
    ids=["served", "other-layer", "all-layers", "unsteered-layer"],
)
def test_unserved_requests_are_rejected(residual_stream, steering, match):
    """A capture or steering layer outside the armed lists, or every layer, raises."""
    config = LensGraphConfig(capture_layers=(2,), steer_layers=(3,))
    vectors = steering and [
        SteeringVector(activations=torch.ones(1, 4), layer_indices=steering)
    ]
    if match is None:
        config.reject_request(residual_stream, vectors)
        return
    with pytest.raises(ValueError, match=match):
        config.reject_request(residual_stream, vectors)


@pytest.mark.parametrize(
    ("hooks", "qk"), [(["hook"], False), ([], True)], ids=["hooks", "qk"]
)
def test_hooks_and_qk_are_rejected(hooks, qk):
    """Hooks and Q/K capture need the forward hooks, so the graph mode refuses them."""
    config = LensGraphConfig(steer_layers=(3,))
    with pytest.raises(ValueError, match="hooks and Q/K"):
        config.reject_hooks(hooks, qk=qk)  # type: ignore[arg-type]
    LensGraphConfig().reject_hooks(hooks, qk=qk)  # type: ignore[arg-type]

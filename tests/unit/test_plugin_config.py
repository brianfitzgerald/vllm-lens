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

    def run(dtype, additional_config=None):
        args = SimpleNamespace(
            worker_extension_cls=None,
            enforce_eager=False,
            additional_config=additional_config,
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
    ("lens", "enforce_eager"),
    [(None, True), ({"capture_layers": [2]}, False), ({"capture_layers": []}, True)],
    ids=["default", "capture-layers", "empty"],
)
def test_a_layer_list_keeps_compilation_on(patched, lens, enforce_eager):
    """A layer list in additional_config leaves enforce_eager off."""
    args = patched(torch.bfloat16, None if lens is None else {CONFIG_KEY: lens})
    assert args.enforce_eager is enforce_eager


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        ({"capture_layers": [5, 2, 2]}, LensGraphConfig(capture_layers=(2, 5))),
        ({"capture_layers": ["3"]}, LensGraphConfig(capture_layers=(3,))),
        ({"capture_layers": ["x"]}, None),
        ({"layers": [1]}, None),
    ],
    ids=["sorted", "string-int", "not-a-layer", "unknown-key"],
)
def test_layer_lists_are_read_and_checked(stored, expected):
    """Layer lists are sorted and deduplicated; anything else raises ValueError."""
    config = SimpleNamespace(additional_config={CONFIG_KEY: stored})
    if expected is None:
        with pytest.raises(ValueError, match="list of layer numbers"):
            LensGraphConfig.from_vllm_config(config)
    else:
        assert LensGraphConfig.from_vllm_config(config) == expected


@pytest.mark.parametrize(
    ("residual_stream", "steering", "match"),
    [
        ("[2]", None, None),
        ("[5]", None, "capture_layers"),
        ("true", None, "must be a list"),
        (None, [2], "apply_steering_vectors"),
    ],
    ids=["served", "other-layer", "all-layers", "steering"],
)
def test_unserved_requests_are_rejected(residual_stream, steering, match):
    """A capture layer outside the list, every layer, or steering raises."""
    config = LensGraphConfig(capture_layers=(2,))
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
    config = LensGraphConfig(capture_layers=(3,))
    with pytest.raises(ValueError, match="hooks and Q/K"):
        config.reject_hooks(hooks, qk=qk)  # type: ignore[arg-type]
    LensGraphConfig().reject_hooks(hooks, qk=qk)  # type: ignore[arg-type]

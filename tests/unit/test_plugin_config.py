"""CPU tests for ``_patched_create_engine_config`` environment defaults."""

from types import SimpleNamespace

import pytest
import torch

import vllm_lens._activations_plugin as plugin
from vllm_lens import SteeringVector
from vllm_lens._cudagraph import CONFIG_KEY, LensGraphConfig


plugin_graph_worker = "vllm_lens._graph_worker.GraphWorker"


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
            worker_cls="auto",
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
    [
        ({"capture_layers": [2]}, False),
        ({"steer_layers": [3]}, False),
        ({"capture_layers": []}, True),
    ],
    ids=["capture-layers", "steer-layers", "empty"],
)
def test_a_layer_list_keeps_compilation_on(patched, lens, enforce_eager):
    """A layer list leaves enforce_eager off and selects the graph worker."""
    args = patched(torch.bfloat16, {CONFIG_KEY: lens})
    assert args.enforce_eager is enforce_eager
    assert args.worker_cls == ("auto" if enforce_eager else plugin_graph_worker)


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        ({"capture_layers": [5, 2, 2]}, LensGraphConfig(capture_layers=(2, 5))),
        (
            {"capture_layers": [2], "steer_layers": [4]},
            LensGraphConfig(capture_layers=(2,), steer_layers=(4,)),
        ),
        (
            {"steer_layers": [1], "steer_norm_match": False},
            LensGraphConfig(steer_layers=(1,), steer_norm_match=False),
        ),
        ({"steer_layers": [1], "steer_norm_match": "no"}, "valid boolean"),
        ({"capture_layers": ["3"]}, "valid integer"),
        ({"capture_layers": [2.0]}, "valid integer"),
        ({"capture_layers": [True]}, "valid integer"),
        ({"capture_layers": [-1]}, "greater than or equal to 0"),
        ({"capture_layers": "12"}, "valid tuple"),
        ({"layers": [1]}, "Extra inputs"),
        ([2], "valid dictionary"),
    ],
    ids=[
        "sorted",
        "both",
        "add-only",
        "norm-not-a-bool",
        "string",
        "float",
        "bool",
        "negative",
        "not-a-list",
        "unknown-key",
        "not-a-dict",
    ],
)
def test_layer_lists_are_read_and_checked(stored, expected):
    """Layer lists are sorted and deduplicated; anything else raises ValueError."""
    config = SimpleNamespace(additional_config={CONFIG_KEY: stored})
    if isinstance(expected, str):
        with pytest.raises(ValueError, match=expected):
            LensGraphConfig.from_vllm_config(config)
    else:
        assert LensGraphConfig.from_vllm_config(config) == expected


@pytest.mark.parametrize(
    ("residual_stream", "steering", "norm_match", "match"),
    [
        ([2], None, False, None),
        ([5], None, False, "capture_layers"),
        (True, None, False, "must be a list"),
        (None, [3], False, None),
        (None, [4], False, "steer_layers"),
        (None, [3], True, "norm_match"),
    ],
    ids=[
        "served",
        "other-layer",
        "all-layers",
        "steered",
        "unsteered-layer",
        "add-only-norm-match",
    ],
)
def test_unserved_requests_are_rejected(residual_stream, steering, norm_match, match):
    """A layer outside the lists, every layer, or norm_match on an add-only engine raises."""
    config = LensGraphConfig(
        capture_layers=(2,), steer_layers=(3,), steer_norm_match=False
    )
    vectors = steering and [
        SteeringVector(
            activations=torch.ones(1, 4), layer_indices=steering, norm_match=norm_match
        )
    ]
    if match is None:
        config.reject(residual_stream=residual_stream, vectors=vectors)
        return
    with pytest.raises(ValueError, match=match):
        config.reject(residual_stream=residual_stream, vectors=vectors)


@pytest.mark.parametrize(
    ("hooks", "qk"), [(["hook"], False), ([], True)], ids=["hooks", "qk"]
)
def test_hooks_and_qk_are_rejected(hooks, qk):
    """Hooks and Q/K capture need the forward hooks, so the graph mode refuses them."""
    config = LensGraphConfig(capture_layers=(3,))
    with pytest.raises(ValueError, match="hooks and Q/K"):
        config.reject(hooks, qk=qk)
    LensGraphConfig().reject(hooks, qk=qk)

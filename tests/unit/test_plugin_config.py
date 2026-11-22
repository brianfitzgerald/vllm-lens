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
        model_config=SimpleNamespace(dtype=dtype), use_v2_model_runner=True
    )


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(
        plugin, "_original_create_engine_config", lambda self, *a, **k: self._cfg
    )
    monkeypatch.delenv("TRITON_F32_DEFAULT", raising=False)
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    monkeypatch.setenv("VLLM_LENS_CAPTURE_DIR", "/tmp")

    def run(dtype, additional_config=None):
        args = SimpleNamespace(
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


@pytest.mark.parametrize(
    "lens",
    [{"capture_layers": [2]}, {"steer_layers": [3]}, {"capture_layers": []}],
    ids=["capture-layers", "steer-layers", "empty"],
)
def test_a_layer_list_selects_the_worker_on_v2(patched, lens):
    """A layer list selects the vllm-lens worker on the V2 runner; with none, the engine
    is left alone."""
    import os

    args = patched(torch.bfloat16, {CONFIG_KEY: lens})
    armed = any(lens.values())
    assert args.worker_cls == (plugin_graph_worker if armed else "auto")
    assert args.enforce_eager is False
    assert os.environ.get("VLLM_USE_V2_MODEL_RUNNER") == ("1" if armed else None)


def test_the_v1_runner_is_refused(patched, monkeypatch):
    """An armed engine that vLLM still builds on the V1 runner raises."""
    monkeypatch.setattr(
        plugin,
        "_original_create_engine_config",
        lambda self, *a, **k: SimpleNamespace(
            model_config=SimpleNamespace(dtype=torch.bfloat16),
            use_v2_model_runner=False,
        ),
    )
    with pytest.raises(RuntimeError, match="V2 model runner"):
        patched(torch.bfloat16, {CONFIG_KEY: {"capture_layers": [1]}})


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


def test_two_position_lists_are_rejected():
    """The vectors of one request share one position list."""
    vectors = [
        SteeringVector(
            activations=torch.ones(1, 4), layer_indices=[3], position_indices=p
        )
        for p in ([1], [2])
    ]
    with pytest.raises(ValueError, match="share position_indices"):
        LensGraphConfig(steer_layers=(3,)).reject(vectors=vectors)


def test_values_per_position_are_rejected():
    """A vector with values per position (3-D activations) is refused."""
    with pytest.raises(ValueError, match="n_layers, hidden_dim"):
        SteeringVector(activations=torch.ones(1, 2, 4), layer_indices=[3])


def test_an_unarmed_server_refuses_lens_requests():
    """With no armed layers, a capture is refused."""
    with pytest.raises(ValueError, match="arms no layers"):
        LensGraphConfig().reject(residual_stream=[0])
    LensGraphConfig().reject()

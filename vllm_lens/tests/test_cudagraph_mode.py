"""Tests for ``VLLM_LENS_CUDAGRAPH``: compilation stays on and hook requests fail."""

import gc
import pytest
import torch
from vllm import LLM, EngineArgs, SamplingParams
from vllm.config.compilation import CUDAGraphMode

from vllm_lens import Hook, SteeringVector
from vllm_lens._cudagraph import CONFIG_KEY, LensGraphConfig

from .conftest import LAYER_IDX, MODEL_NAME, PROMPT


@pytest.fixture(scope="module")
def graph_llm():
    """Offline engine started with ``VLLM_LENS_CUDAGRAPH=1``."""
    # The variable is read once, when the engine config is created.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("VLLM_LENS_CUDAGRAPH", "1")
        engine = LLM(model=MODEL_NAME, dtype="auto", gpu_memory_utilization=0.3)
    yield engine
    del engine
    gc.collect()
    torch.cuda.empty_cache()


def test_eager_mode_is_not_forced(graph_llm):
    """The engine keeps the vLLM defaults: not eager, with CUDA graphs on."""
    vllm_config = graph_llm.llm_engine.vllm_config
    assert vllm_config.model_config.enforce_eager is False
    assert vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE


def test_persistent_hook_registration_is_rejected(graph_llm):
    """A persistent hook would never run, so its registration raises."""
    hook = Hook(fn=lambda ctx, hidden: None, layer_indices=[LAYER_IDX])
    with pytest.raises(ValueError, match="VLLM_LENS_CUDAGRAPH"):
        graph_llm.register_hooks([hook])


def test_plain_generation_works(graph_llm):
    """A request with no vllm-lens option is served."""
    outputs = graph_llm.generate([PROMPT], SamplingParams(max_tokens=4))
    assert outputs[0].outputs[0].token_ids


@pytest.mark.parametrize(
    "extra_args",
    [
        {"output_residual_stream": [LAYER_IDX]},
        {
            "apply_steering_vectors": [
                SteeringVector(activations=torch.zeros(1, 8), layer_indices=[LAYER_IDX])
            ]
        },
        {"output_qk": [LAYER_IDX]},
    ],
    ids=["capture", "steering", "qk"],
)
def test_hook_requests_are_rejected(graph_llm, extra_args):
    """Capture, steering and Q/K capture need the forward hooks, so they raise."""
    params = SamplingParams(max_tokens=1, extra_args=extra_args)
    with pytest.raises(ValueError, match="VLLM_LENS_CUDAGRAPH"):
        graph_llm.generate([PROMPT], params)


def test_a_reused_engine_args_follows_the_environment(monkeypatch):
    """The second config of one ``EngineArgs`` keeps nothing from the first. No engine."""
    engine_args = EngineArgs(model=MODEL_NAME)
    monkeypatch.setenv("VLLM_LENS_CUDAGRAPH", "1")
    graph = engine_args.create_engine_config()
    monkeypatch.delenv("VLLM_LENS_CUDAGRAPH")
    eager = engine_args.create_engine_config()

    assert graph.model_config.enforce_eager is False
    assert LensGraphConfig.from_vllm_config(graph).enabled
    assert eager.model_config.enforce_eager is True
    assert CONFIG_KEY not in eager.additional_config

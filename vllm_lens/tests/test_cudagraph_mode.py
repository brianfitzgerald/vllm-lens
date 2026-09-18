"""Tests for ``VLLM_LENS_CUDAGRAPH``: compilation stays on and hook requests fail."""

import gc

import pytest
import torch
from vllm import LLM, SamplingParams

from vllm_lens import SteeringVector

from .conftest import LAYER_IDX, MODEL_NAME, PROMPT


@pytest.fixture(scope="module")
def graph_llm():
    """Offline engine started with ``VLLM_LENS_CUDAGRAPH=1``."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("VLLM_LENS_CUDAGRAPH", "1")
        engine = LLM(model=MODEL_NAME, dtype="auto", gpu_memory_utilization=0.3)
        yield engine
        del engine
    gc.collect()
    torch.cuda.empty_cache()


def test_eager_mode_is_not_forced(graph_llm):
    """The engine keeps the vLLM default, which is not eager."""
    assert graph_llm.llm_engine.vllm_config.model_config.enforce_eager is False


def test_plain_generation_works(graph_llm):
    """A request with no vllm-lens option is served."""
    outputs = graph_llm.generate([PROMPT], SamplingParams(max_tokens=4))
    assert len(outputs[0].outputs[0].token_ids) == 4


@pytest.mark.parametrize(
    ("extra_args", "error"),
    [
        ({"output_residual_stream": [LAYER_IDX]}, ValueError),
        (
            {
                "apply_steering_vectors": [
                    SteeringVector(
                        activations=torch.zeros(1, 8), layer_indices=[LAYER_IDX]
                    )
                ]
            },
            RuntimeError,
        ),
    ],
    ids=["capture", "steering"],
)
def test_unserved_requests_are_rejected(graph_llm, extra_args, error):
    """No capture layer is armed and steering needs the forward hooks, so both raise."""
    params = SamplingParams(max_tokens=1, extra_args=extra_args)
    with pytest.raises(error, match="VLLM_LENS_CUDAGRAPH"):
        graph_llm.generate([PROMPT], params)

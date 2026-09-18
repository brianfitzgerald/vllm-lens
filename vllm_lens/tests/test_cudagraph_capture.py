"""Tests for capture under CUDA graphs via ``VLLM_LENS_CAPTURE_LAYERS``."""

import gc

import pytest
import torch
from vllm import LLM, SamplingParams

from .conftest import MODEL_NAME, NUM_LAYERS, PROMPT

LONG_PROMPT = "The quick brown fox jumps over the lazy dog. " * 20
ARMED_LAYERS = [0, 2, NUM_LAYERS - 1]


def _make_llm(env: dict[str, str], **kwargs):
    """Start an offline engine with ``env`` set, so its workers inherit it."""
    with pytest.MonkeyPatch.context() as patch:
        for name, value in env.items():
            patch.setenv(name, value)
        return LLM(model=MODEL_NAME, dtype="auto", gpu_memory_utilization=0.2, **kwargs)


@pytest.fixture(scope="module")
def eager_llm():
    """The reference: forward hooks in eager mode."""
    engine = _make_llm({})
    yield engine
    del engine
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def aux_eager_llm():
    """Aux capture with ``enforce_eager``: the kernels of ``eager_llm``."""
    env = {"VLLM_LENS_CAPTURE_LAYERS": ",".join(map(str, ARMED_LAYERS))}
    engine = _make_llm(env, enforce_eager=True)
    yield engine
    del engine
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module", params=[None, 64], ids=["single-pass", "chunked-64"])
def graph_llm(request):
    """CUDA graphs on, with ``ARMED_LAYERS`` captured as auxiliary hidden states."""
    kwargs = {}
    if request.param is not None:
        kwargs = {
            "max_num_batched_tokens": request.param,
            "enable_chunked_prefill": True,
        }
    env = {"VLLM_LENS_CAPTURE_LAYERS": ",".join(map(str, ARMED_LAYERS))}
    engine = _make_llm(env, **kwargs)
    yield engine
    del engine
    gc.collect()
    torch.cuda.empty_cache()


def _capture(engine: LLM, layers: list[int], pool: str | None, max_tokens: int):
    """Capture ``layers`` for the short and the long prompt."""
    extra_args: dict = {"output_residual_stream": layers}
    if pool is not None:
        extra_args["output_residual_stream_pool"] = pool
    params = SamplingParams(
        temperature=0.0, max_tokens=max_tokens, extra_args=extra_args
    )
    outputs = engine.generate([PROMPT, LONG_PROMPT], params)
    return [out.activations["residual_stream"] for out in outputs]  # type: ignore[reportAttributeAccessIssue]


def test_compilation_is_on(graph_llm):
    """A capture layer list alone turns the CUDA-graph mode on."""
    assert graph_llm.llm_engine.vllm_config.model_config.enforce_eager is False


def _relative_errors(actual: torch.Tensor, expected: torch.Tensor) -> list[float]:
    """``||actual - expected|| / ||expected||`` for each layer, in float32."""
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    diff = (actual.float() - expected.float()).flatten(1).norm(dim=1)
    return (diff / expected.float().flatten(1).norm(dim=1)).tolist()


@pytest.mark.parametrize("pool", [None, "last", "mean"])
def test_aux_capture_matches_the_forward_hooks(eager_llm, aux_eager_llm, pool):
    """With the same kernels, aux capture returns the rows the forward hooks return."""
    hooks = _capture(eager_llm, ARMED_LAYERS, pool, 4)
    aux = _capture(aux_eager_llm, ARMED_LAYERS, pool, 4)
    for hook_acts, aux_acts in zip(hooks, aux):
        errors = _relative_errors(aux_acts, hook_acts)
        assert max(errors) < 1e-4, f"relative error per layer: {errors}"


@pytest.mark.parametrize("max_tokens", [1, 4])
@pytest.mark.parametrize("pool", [None, "last", "mean"])
def test_graph_capture_matches_eager_capture(eager_llm, graph_llm, pool, max_tokens):
    """Compiled kernels differ from eager ones, so this tolerance is loose."""
    eager = _capture(eager_llm, ARMED_LAYERS, pool, max_tokens)
    graph = _capture(graph_llm, ARMED_LAYERS, pool, max_tokens)
    for eager_acts, graph_acts in zip(eager, graph):
        errors = _relative_errors(graph_acts, eager_acts)
        assert max(errors) < 5e-2, f"relative error per layer: {errors}"


def test_a_subset_of_the_armed_layers(graph_llm):
    """A request for one armed layer gets that layer alone."""
    acts = _capture(graph_llm, [2], None, 1)
    assert acts[0].shape[0] == 1


def test_an_unarmed_layer_is_rejected(graph_llm):
    """A layer outside ``VLLM_LENS_CAPTURE_LAYERS`` raises before generation."""
    with pytest.raises(ValueError, match="VLLM_LENS_CAPTURE_LAYERS"):
        _capture(graph_llm, [1], None, 1)

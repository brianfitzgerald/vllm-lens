"""Tests for capture under CUDA graphs via ``VLLM_LENS_CAPTURE_LAYERS``."""

import gc

import pytest
import torch
from vllm import LLM, SamplingParams

from .conftest import NUM_LAYERS, PROMPT, make_llm

LONG_PROMPT = "The quick brown fox jumps over the lazy dog. " * 20
CAPTURE_LAYERS = [0, 2, NUM_LAYERS - 1]


@pytest.fixture(scope="module")
def aux_eager_llm():
    """Aux capture with ``enforce_eager``: the kernels of ``eager_llm``."""
    env = {"VLLM_LENS_CAPTURE_LAYERS": ",".join(map(str, CAPTURE_LAYERS))}
    engine = make_llm(env, enforce_eager=True)
    yield engine
    del engine
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module", params=[None, 64], ids=["single-pass", "chunked-64"])
def graph_llm(request):
    """CUDA graphs on, with ``CAPTURE_LAYERS`` captured as auxiliary hidden states."""
    kwargs = {}
    if request.param is not None:
        kwargs = {
            "max_num_batched_tokens": request.param,
            "enable_chunked_prefill": True,
        }
    env = {"VLLM_LENS_CAPTURE_LAYERS": ",".join(map(str, CAPTURE_LAYERS))}
    engine = make_llm(env, **kwargs)
    yield engine
    del engine
    gc.collect()
    torch.cuda.empty_cache()


def _capture(engine: LLM, layers: list[int] | bool, pool: str | None, max_tokens: int):
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
    hooks = _capture(eager_llm, CAPTURE_LAYERS, pool, 4)
    aux = _capture(aux_eager_llm, CAPTURE_LAYERS, pool, 4)
    for hook_acts, aux_acts in zip(hooks, aux):
        errors = _relative_errors(aux_acts, hook_acts)
        # Both engines run the eager kernels, so only the capture path differs.
        assert max(errors) < 1e-4, f"relative error per layer: {errors}"


@pytest.mark.parametrize("max_tokens", [1, 4])
@pytest.mark.parametrize("pool", [None, "last", "mean"])
def test_graph_capture_matches_eager_capture(eager_llm, graph_llm, pool, max_tokens):
    """Compiled kernels differ from eager ones, so the limit is 5% per layer."""
    eager = _capture(eager_llm, CAPTURE_LAYERS, pool, max_tokens)
    graph = _capture(graph_llm, CAPTURE_LAYERS, pool, max_tokens)
    for eager_acts, graph_acts in zip(eager, graph):
        errors = _relative_errors(graph_acts, eager_acts)
        assert max(errors) < 5e-2, f"relative error per layer: {errors}"


def test_a_subset_of_the_capture_layers(eager_llm, graph_llm):
    """A request for one of the named layers gets the rows of that layer alone."""
    eager = _capture(eager_llm, [2], None, 1)
    graph = _capture(graph_llm, [2], None, 1)
    for eager_acts, graph_acts in zip(eager, graph):
        assert max(_relative_errors(graph_acts, eager_acts)) < 5e-2


@pytest.mark.parametrize("layers", [[1], True], ids=["other-layer", "all-layers"])
def test_an_unserved_capture_is_rejected(graph_llm, layers):
    """A layer outside the list, or "all layers", raises before generation."""
    with pytest.raises(ValueError, match="VLLM_LENS_CAPTURE_LAYERS"):
        _capture(graph_llm, layers, None, 1)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="PP=2 requires at least 2 GPUs"
)
def test_pipeline_parallelism_is_refused_at_load():
    """Only the last pipeline stage returns aux hidden states, so the load fails.
    The parent sees only the failed start; the reason is in the worker log."""
    with pytest.raises(
        Exception, match="(?i)initialization failed|pipeline parallelism"
    ):
        make_llm({"VLLM_LENS_CAPTURE_LAYERS": "2"}, pipeline_parallel_size=2)

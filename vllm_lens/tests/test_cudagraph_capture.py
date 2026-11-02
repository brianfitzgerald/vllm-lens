"""Tests for capture under CUDA graphs via ``additional_config["vllm_lens"]``."""

import pytest
import torch
from vllm import LLM, SamplingParams

from .conftest import LONG_PROMPT, NUM_LAYERS, PROMPT, lens_engine, relative_errors

GRAPH_LAYERS = [0, 2, NUM_LAYERS - 1]


ENGINES = {
    # The lens op with enforce_eager: the kernels of eager_llm.
    "op-eager": {"enforce_eager": True, "enable_prefix_caching": False},
    "graph": {},
    "graph-chunked-64": {"max_num_batched_tokens": 64, "enable_chunked_prefill": True},
}


@pytest.fixture(scope="module", params=ENGINES)
def lens_llm(request):
    """``GRAPH_LAYERS`` captured by the lens op, with eager kernels or CUDA graphs."""
    yield from lens_engine({"capture_layers": GRAPH_LAYERS}, **ENGINES[request.param])


def _capture(engine: LLM):
    """Capture ``GRAPH_LAYERS`` for the short and the long prompt."""
    params = SamplingParams(
        temperature=0.0,
        max_tokens=4,
        extra_args={"output_residual_stream": GRAPH_LAYERS},
    )
    outputs = engine.generate([PROMPT, LONG_PROMPT], params)
    return [out.activations["residual_stream"] for out in outputs]  # type: ignore[reportAttributeAccessIssue]


def test_op_capture_matches_the_forward_hooks(eager_llm, lens_llm):
    """The lens op returns the rows the forward hooks return: with the same kernels at
    1e-4, and with compiled kernels, which differ, within 5% per layer."""
    eager = lens_llm.llm_engine.vllm_config.model_config.enforce_eager
    tolerance = 1e-4 if eager else 5e-2
    for hook_acts, op_acts in zip(_capture(eager_llm), _capture(lens_llm)):
        errors = relative_errors(op_acts, hook_acts).tolist()
        assert max(errors) < tolerance, f"relative error per layer: {errors}"


@pytest.mark.parametrize("fused", [False, True], ids=["plain", "fused-residual"])
def test_the_capture_op_copies_the_stream(fused):
    """The op writes ``hidden + residual`` into the first rows of the buffer. No GPU."""
    from vllm_lens._graph_worker import _capture_op

    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(3, 8, generator=generator)
    residual = torch.randn(3, 8, generator=generator) if fused else None
    out = torch.zeros(5, 8)
    _capture_op(hidden, residual, out)
    expected = hidden if residual is None else hidden + residual
    assert torch.equal(out[:3], expected) and not out[3:].any()

"""Tests for capture under CUDA graphs via ``additional_config["vllm_lens"]``."""

import pytest
import torch
from vllm import LLM, SamplingParams

from .conftest import LONG_PROMPT, NUM_LAYERS, PROMPT, lens_engine, relative_errors

GRAPH_LAYERS = [0, 2, NUM_LAYERS - 1]


ENGINES = {
    "graph": {},
    "graph-chunked-64": {"max_num_batched_tokens": 64, "enable_chunked_prefill": True},
}


@pytest.fixture(scope="module", params=ENGINES)
def lens_llm(request):
    """``GRAPH_LAYERS`` captured under CUDA graphs, in one pass or 64-token chunks."""
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


def test_graph_capture_matches_eager_capture(eager_llm, lens_llm):
    """Graph capture returns the rows of eager capture within 5% per layer: compiled
    kernels differ from eager ones."""
    for eager_acts, graph_acts in zip(_capture(eager_llm), _capture(lens_llm)):
        errors = relative_errors(graph_acts, eager_acts).tolist()
        assert max(errors) < 5e-2, f"relative error per layer: {errors}"


@pytest.mark.parametrize("fused", [False, True], ids=["plain", "fused-residual"])
def test_the_capture_op_writes_ring_rows(fused):
    """The op writes ``hidden + residual`` to each token's ring row; padding tokens go to
    the sentinel row. No GPU."""
    from vllm_lens._graph_worker import _capture_op

    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(4, 8, generator=generator)
    residual = torch.randn(4, 8, generator=generator) if fused else None
    ring = torch.zeros(6, 8)
    dest = torch.tensor([3, 0, 5, 5, 5, 5])
    _capture_op(hidden, residual, ring, dest)
    expected = hidden if residual is None else hidden + residual
    assert torch.equal(ring[3], expected[0]) and torch.equal(ring[0], expected[1])
    assert not ring[[1, 2, 4]].any()

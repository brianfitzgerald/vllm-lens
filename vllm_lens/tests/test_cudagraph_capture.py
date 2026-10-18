"""Tests for capture under CUDA graphs via ``additional_config["vllm_lens"]``."""

import gc

import pytest
import torch
from vllm import LLM, SamplingParams

from vllm_lens import SteeringVector

from .conftest import LONG_PROMPT, NUM_LAYERS, PROMPT, make_llm

GRAPH_LAYERS = [0, 2, NUM_LAYERS - 1]


@pytest.fixture(scope="module")
def op_eager_llm():
    """The lens op with ``enforce_eager``: the kernels of ``eager_llm``."""
    env = {"capture_layers": GRAPH_LAYERS}
    engine = make_llm(env, enforce_eager=True, enable_prefix_caching=False)
    yield engine
    del engine
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module", params=[None, 64], ids=["single-pass", "chunked-64"])
def graph_llm(request):
    """CUDA graphs on, with ``GRAPH_LAYERS`` captured by the lens op."""
    kwargs = {}
    if request.param is not None:
        kwargs = {
            "max_num_batched_tokens": request.param,
            "enable_chunked_prefill": True,
        }
    env = {"capture_layers": GRAPH_LAYERS}
    engine = make_llm(env, **kwargs)
    yield engine
    del engine
    gc.collect()
    torch.cuda.empty_cache()


def _capture(engine: LLM, layers: list[int] | bool, pool: str | None, max_tokens: int):
    """Capture ``layers`` for the short and the long prompt. ``pool="last"`` selects the
    last prompt token; with ``pool="project"`` each position is its projection on two fixed
    directions and its norm."""
    extra_args: dict = {"output_residual_stream": layers}
    if pool == "project":
        hidden = engine.llm_engine.vllm_config.model_config.get_hidden_size()
        directions = torch.randn(2, hidden, generator=torch.Generator().manual_seed(0))
        engine.set_projection("fixed", directions)  # type: ignore[attr-defined]
        extra_args["output_residual_stream_project"] = "fixed"
    elif pool == "last":
        extra_args["output_residual_stream_positions"] = "last_prompt_token"
    elif pool is not None:
        extra_args["output_residual_stream_pool"] = pool
    params = SamplingParams(
        temperature=0.0, max_tokens=max_tokens, extra_args=extra_args
    )
    outputs = engine.generate([PROMPT, LONG_PROMPT], params)
    if pool == "project":
        return [
            torch.cat(
                [
                    out.activations["residual_stream_projection"],  # type: ignore[reportAttributeAccessIssue]
                    out.activations["residual_stream_norm"].unsqueeze(-1),  # type: ignore[reportAttributeAccessIssue]
                ],
                dim=-1,
            )
            for out in outputs
        ]
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


@pytest.mark.parametrize("pool", [None, "last", "mean", "project"])
def test_op_capture_matches_the_forward_hooks(eager_llm, op_eager_llm, pool):
    """With the same kernels, the lens op returns the rows the forward hooks return."""
    hooks = _capture(eager_llm, GRAPH_LAYERS, pool, 4)
    op = _capture(op_eager_llm, GRAPH_LAYERS, pool, 4)
    for hook_acts, op_acts in zip(hooks, op):
        errors = _relative_errors(op_acts, hook_acts)
        # Both engines run the eager kernels, so only the capture path differs.
        assert max(errors) < 1e-4, f"relative error per layer: {errors}"


@pytest.mark.parametrize(
    ("layers", "max_tokens"),
    [(GRAPH_LAYERS, 1), (GRAPH_LAYERS, 4), ([2], 1)],
    ids=["all-1", "all-4", "subset-1"],
)
def test_graph_capture_matches_eager_capture(eager_llm, graph_llm, layers, max_tokens):
    """Compiled kernels differ from eager ones, so the limit is 5% per layer. Pooling and
    projection share the capture path, which the op test compares at 1e-4."""
    eager = _capture(eager_llm, layers, None, max_tokens)
    graph = _capture(graph_llm, layers, None, max_tokens)
    for eager_acts, graph_acts in zip(eager, graph):
        errors = _relative_errors(graph_acts, eager_acts)
        assert max(errors) < 5e-2, f"relative error per layer: {errors}"


@pytest.mark.parametrize("max_tokens", [1, 4])
def test_graph_projection_matches_the_graph_capture(graph_llm, max_tokens):
    """Under CUDA graphs the projected capture is the projection of the full capture."""
    hidden = graph_llm.llm_engine.vllm_config.model_config.get_hidden_size()
    directions = torch.randn(2, hidden, generator=torch.Generator().manual_seed(0))
    full = _capture(graph_llm, GRAPH_LAYERS, None, max_tokens)
    projected = _capture(graph_llm, GRAPH_LAYERS, "project", max_tokens)
    for full_acts, projected_acts in zip(full, projected):
        rows = full_acts.float()
        expected = torch.cat(
            [rows @ directions.T, rows.norm(dim=-1, keepdim=True)], dim=-1
        )
        errors = _relative_errors(projected_acts, expected)
        assert max(errors) < 1e-4, f"relative error per layer: {errors}"


@pytest.mark.parametrize("layers", [[1], True], ids=["other-layer", "all-layers"])
def test_an_unserved_capture_is_rejected(graph_llm, layers):
    """A layer outside the list, or "all layers", raises before generation."""
    with pytest.raises(ValueError, match="capture_layers"):
        _capture(graph_llm, layers, None, 1)


def test_steering_is_rejected(graph_llm):
    """A capture-only server refuses steering vectors before generation."""
    vector = SteeringVector(activations=torch.ones(1, 896), layer_indices=[2])
    params = SamplingParams(
        max_tokens=1, extra_args={"apply_steering_vectors": [vector]}
    )
    with pytest.raises(ValueError, match="apply_steering_vectors"):
        graph_llm.generate([PROMPT], params)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("fused", [False, True], ids=["plain", "fused-residual"])
def test_the_capture_op_copies_the_stream(dtype, fused):
    """The op writes ``hidden + residual`` into the first rows of the buffer. No GPU."""
    import vllm_lens._graph_worker  # noqa: F401

    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(3, 8, generator=generator).to(dtype)
    residual = torch.randn(3, 8, generator=generator).to(dtype) if fused else None
    out = torch.zeros(5, 8, dtype=dtype)
    torch.ops.vllm_lens.capture(hidden, residual, out)
    expected = hidden if residual is None else hidden + residual
    assert torch.equal(out[:3], expected) and not out[3:].any()
    with pytest.raises(RuntimeError, match="max_num_batched_tokens"):
        torch.ops.vllm_lens.capture(torch.zeros(6, 8, dtype=dtype), None, out)

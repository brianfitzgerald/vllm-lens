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


def _capture(engine: LLM, layers: list[int], pool: str | None, max_tokens: int):
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


@pytest.mark.parametrize(
    ("layers", "pool"),
    [
        (GRAPH_LAYERS, None),
        ([2], None),
        (GRAPH_LAYERS, "last"),
        (GRAPH_LAYERS, "mean"),
        (GRAPH_LAYERS, "project"),
    ],
    ids=["all", "subset", "last", "mean", "project"],
)
def test_op_capture_matches_the_forward_hooks(eager_llm, lens_llm, layers, pool):
    """The lens op returns the rows the forward hooks return: with the same kernels at
    1e-4, and with compiled kernels, which differ, within 5% per layer."""
    eager = lens_llm.llm_engine.vllm_config.model_config.enforce_eager
    tolerance = 1e-4 if eager else 5e-2
    hooks = _capture(eager_llm, layers, pool, 4)
    op = _capture(lens_llm, layers, pool, 4)
    for hook_acts, op_acts in zip(hooks, op):
        errors = relative_errors(op_acts, hook_acts).tolist()
        assert max(errors) < tolerance, f"relative error per layer: {errors}"


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

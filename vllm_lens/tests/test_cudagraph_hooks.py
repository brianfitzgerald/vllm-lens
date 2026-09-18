"""Tests for hooks under CUDA graphs via ``VLLM_LENS_HOOK_LAYERS``."""

import gc

import pytest
import torch
from vllm import LLM, SamplingParams

from vllm_lens import Hook, SteeringVector

from .conftest import make_llm, prompt_rows, row_error

LONG_PROMPT = "The quick brown fox jumps over the lazy dog. " * 20
HOOK_LAYER = 2


@pytest.fixture(scope="module")
def graph_llm():
    """CUDA graphs on, 64-token prefill chunks, the split hook on one layer."""
    engine = make_llm(
        {"VLLM_LENS_HOOK_LAYERS": str(HOOK_LAYER)},
        max_num_batched_tokens=64,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
    )
    yield engine
    del engine
    gc.collect()
    torch.cuda.empty_cache()


def _hooks(kind: str, layer: int = HOOK_LAYER, pre: bool = False) -> list[Hook]:
    """One hook. Its function is local, so cloudpickle sends it by value."""

    def save_rows(ctx, hidden):
        ctx.saved.setdefault("rows", []).append(hidden.float().cpu())

    def add_one(ctx, hidden):
        return hidden + 1.0

    fn = {"save": save_rows, "add": add_one}[kind]
    return [Hook(fn=fn, layer_indices=[layer], pre=pre)]


def _run(engine: LLM, extra_args: dict, max_tokens: int = 4):
    """Generate ``LONG_PROMPT`` with ``extra_args`` plus capture on ``HOOK_LAYER``."""
    params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        extra_args={"output_residual_stream": [HOOK_LAYER], **extra_args},
    )
    return engine.generate([LONG_PROMPT], params)[0]


def _vector(hidden: int, **kwargs) -> SteeringVector:
    """A reproducible random vector on ``HOOK_LAYER``."""
    activations = torch.randn(1, hidden, generator=torch.Generator().manual_seed(0))
    return SteeringVector(activations=activations, layer_indices=[HOOK_LAYER], **kwargs)


EXTRA_ARGS = {
    "capture": lambda hidden: {},
    "steering": lambda hidden: {
        "apply_steering_vectors": [_vector(hidden, scale=4.0, norm_match=True)]
    },
    "modifying-hook": lambda hidden: {"apply_hooks": _hooks("add")},
}


def test_the_hook_op_is_a_piecewise_split_point(graph_llm):
    """Compilation is on, in PIECEWISE mode, with the hook op in ``splitting_ops``."""
    from vllm.config.compilation import CUDAGraphMode

    vllm_config = graph_llm.llm_engine.vllm_config
    assert vllm_config.model_config.enforce_eager is False
    assert vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE
    assert "vllm_lens::hook" in vllm_config.compilation_config.splitting_ops


@pytest.mark.parametrize("case", EXTRA_ARGS)
def test_split_hook_matches_the_eager_hooks(eager_llm, graph_llm, hidden, case):
    """Capture, steering and a modifying hook give the eager stream, row by row."""
    eager = _run(eager_llm, EXTRA_ARGS[case](hidden))
    graph = _run(graph_llm, EXTRA_ARGS[case](hidden))
    error = row_error(prompt_rows(graph), prompt_rows(eager))
    assert error < 5e-2, f"largest row error {error:.4f}"


def test_a_capture_hook_sees_the_eager_rows(eager_llm, graph_llm):
    """``ctx.saved`` under CUDA graphs holds the rows eager capture returns, under key 0."""
    eager = _run(eager_llm, {})
    graph = _run(graph_llm, {"apply_hooks": _hooks("save")})
    saved = torch.cat(graph.hook_results["0"]["rows"])  # type: ignore[reportAttributeAccessIssue]
    n_prompt = len(eager.prompt_token_ids)
    assert row_error(saved[:n_prompt], prompt_rows(eager)) < 5e-2


def test_persistent_and_request_hooks_keep_their_results_apart(graph_llm):
    """A persistent hook does not shift the result key of a per-request hook."""
    graph_llm.register_hooks(_hooks("save"))
    try:
        output = _run(graph_llm, {"apply_hooks": _hooks("save")})
        assert list(output.hook_results) == ["0"]  # type: ignore[reportAttributeAccessIssue]
        assert graph_llm.collect_hook_results()
    finally:
        graph_llm.clear_hooks()


@pytest.mark.parametrize(
    "kwargs",
    [{"pre": True}, {"layer": HOOK_LAYER + 1}],
    ids=["pre-hook", "other-layer"],
)
def test_unserved_hooks_are_rejected(graph_llm, kwargs):
    """A pre-hook, or a hook on another layer, raises before generation."""
    with pytest.raises(ValueError, match="VLLM_LENS_CUDAGRAPH"):
        _run(graph_llm, {"apply_hooks": _hooks("save", **kwargs)})
    with pytest.raises(ValueError, match="VLLM_LENS_CUDAGRAPH"):
        graph_llm.register_hooks(_hooks("save", **kwargs))


def test_hook_layers_with_enforce_eager_are_refused():
    """No piecewise attention split exists in eager mode, so the config raises."""
    with pytest.raises(ValueError, match="VLLM_LENS_HOOK_LAYERS"):
        make_llm({"VLLM_LENS_HOOK_LAYERS": str(HOOK_LAYER)}, enforce_eager=True)


def test_a_layer_in_two_lists_is_rejected(monkeypatch):
    """A hook layer that is also a capture layer would be captured twice."""
    from vllm_lens._cudagraph import LensGraphConfig

    monkeypatch.setenv("VLLM_LENS_HOOK_LAYERS", "2,5")
    monkeypatch.setenv("VLLM_LENS_CAPTURE_LAYERS", "5")
    with pytest.raises(ValueError, match="name each layer once"):
        LensGraphConfig.from_env()

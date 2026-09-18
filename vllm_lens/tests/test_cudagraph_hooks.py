"""Tests for hooks under CUDA graphs via ``VLLM_LENS_HOOK_LAYERS``."""

import gc

import pytest
import torch
from vllm import LLM, SamplingParams

from vllm_lens import Hook, SteeringVector

from .conftest import MODEL_NAME

LONG_PROMPT = "The quick brown fox jumps over the lazy dog. " * 20
HOOK_LAYER = 2
HIDDEN = 896


def _make_llm(env: dict[str, str], **kwargs):
    """Start an offline engine with ``env`` set, so its workers inherit it."""
    with pytest.MonkeyPatch.context() as patch:
        for name, value in env.items():
            patch.setenv(name, value)
        return LLM(
            model=MODEL_NAME,
            dtype="auto",
            gpu_memory_utilization=0.2,
            enable_prefix_caching=False,
            **kwargs,
        )


@pytest.fixture(scope="module")
def eager_llm():
    """The reference: forward hooks in eager mode."""
    engine = _make_llm({})
    yield engine
    del engine
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def graph_llm():
    """CUDA graphs on, 64-token prefill chunks, the split hook on one layer."""
    engine = _make_llm(
        {"VLLM_LENS_HOOK_LAYERS": str(HOOK_LAYER)},
        max_num_batched_tokens=64,
        enable_chunked_prefill=True,
    )
    yield engine
    del engine
    gc.collect()
    torch.cuda.empty_cache()


def _save_rows(ctx, hidden):
    """Capture hook: append the rows of this step."""
    ctx.saved.setdefault("rows", []).append(hidden.float().cpu())


def _add_one(ctx, hidden):
    """Modifying hook: shift the stream by a constant."""
    return hidden + 1.0


def _run(engine: LLM, extra_args: dict, max_tokens: int = 4):
    """Generate ``LONG_PROMPT`` with ``extra_args`` plus capture on ``HOOK_LAYER``."""
    params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        extra_args={"output_residual_stream": [HOOK_LAYER], **extra_args},
    )
    return engine.generate([LONG_PROMPT], params)[0]


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """``||actual - expected|| / ||expected||`` in float32."""
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


def _vector(**kwargs) -> SteeringVector:
    """A reproducible random vector on ``HOOK_LAYER``."""
    activations = torch.randn(1, HIDDEN, generator=torch.Generator().manual_seed(0))
    return SteeringVector(activations=activations, layer_indices=[HOOK_LAYER], **kwargs)


EXTRA_ARGS = {
    "capture": lambda: {},
    "steering": lambda: {
        "apply_steering_vectors": [_vector(scale=4.0, norm_match=True)]
    },
    "modifying-hook": lambda: {
        "apply_hooks": [Hook(fn=_add_one, layer_indices=[HOOK_LAYER])]
    },
}


def test_the_hook_op_is_a_piecewise_split_point(graph_llm):
    """Compilation is on, in PIECEWISE mode, with the hook op in ``splitting_ops``."""
    from vllm.config.compilation import CUDAGraphMode

    vllm_config = graph_llm.llm_engine.vllm_config
    assert vllm_config.model_config.enforce_eager is False
    assert vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE
    assert "vllm_lens::hook" in vllm_config.compilation_config.splitting_ops


@pytest.mark.parametrize("case", EXTRA_ARGS)
def test_split_hook_matches_the_eager_hooks(eager_llm, graph_llm, case):
    """Capture, steering and a modifying hook give the eager stream."""
    eager = _run(eager_llm, EXTRA_ARGS[case]())
    graph = _run(graph_llm, EXTRA_ARGS[case]())
    n_prompt = len(eager.prompt_token_ids)
    eager_acts = eager.activations["residual_stream"][:, :n_prompt]  # type: ignore[reportAttributeAccessIssue]
    graph_acts = graph.activations["residual_stream"][:, :n_prompt]  # type: ignore[reportAttributeAccessIssue]
    error = _relative_error(graph_acts, eager_acts)
    assert error < 5e-2, f"relative error {error:.4f}"


def test_a_capture_hook_sees_the_captured_rows(graph_llm):
    """``ctx.saved`` holds the rows that native capture returns, under result key 0."""
    hook = Hook(fn=_save_rows, layer_indices=[HOOK_LAYER])
    output = _run(graph_llm, {"apply_hooks": [hook]})
    saved = torch.cat(output.hook_results["0"]["rows"])  # type: ignore[reportAttributeAccessIssue]
    captured = output.activations["residual_stream"][0]  # type: ignore[reportAttributeAccessIssue]
    assert _relative_error(saved[: captured.shape[0]], captured) < 1e-3


def test_persistent_and_request_hooks_keep_their_results_apart(graph_llm):
    """A persistent hook does not shift the result key of a per-request hook."""
    graph_llm.register_hooks([Hook(fn=_save_rows, layer_indices=[HOOK_LAYER])])
    try:
        hook = Hook(fn=_save_rows, layer_indices=[HOOK_LAYER])
        output = _run(graph_llm, {"apply_hooks": [hook]})
        assert list(output.hook_results) == ["0"]  # type: ignore[reportAttributeAccessIssue]
        assert graph_llm.collect_hook_results()
    finally:
        graph_llm.clear_hooks()


@pytest.mark.parametrize(
    "hook",
    [
        Hook(fn=_save_rows, layer_indices=[HOOK_LAYER], pre=True),
        Hook(fn=_save_rows, layer_indices=[HOOK_LAYER + 1]),
    ],
    ids=["pre-hook", "unarmed-layer"],
)
def test_unserved_hooks_are_rejected(graph_llm, hook):
    """A pre-hook, or a hook on another layer, raises before generation."""
    with pytest.raises(ValueError, match="VLLM_LENS_CUDAGRAPH"):
        _run(graph_llm, {"apply_hooks": [hook]})
    with pytest.raises(ValueError, match="VLLM_LENS_CUDAGRAPH"):
        graph_llm.register_hooks([hook])


def test_a_layer_in_two_lists_is_rejected(monkeypatch):
    """A hook layer that is also a capture layer would be captured twice."""
    from vllm_lens._cudagraph import LensGraphConfig

    monkeypatch.setenv("VLLM_LENS_HOOK_LAYERS", "2,5")
    monkeypatch.setenv("VLLM_LENS_CAPTURE_LAYERS", "5")
    with pytest.raises(ValueError, match="name each layer once"):
        LensGraphConfig.from_env()

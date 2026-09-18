"""Tests for pooled capture via ``output_residual_stream_pool``."""

import gc

import pytest
import torch
from vllm import LLM, AsyncEngineArgs, AsyncLLMEngine, SamplingParams

from .conftest import LAYER_IDX, MODEL_NAME, PROMPT

LONG_PROMPT = "The quick brown fox jumps over the lazy dog. " * 20
LAYERS = [LAYER_IDX, LAYER_IDX + 3]


@pytest.fixture(scope="module", params=[None, 64], ids=["single-pass", "chunked-64"])
def llm(request):
    """Offline engine, with and without a 64-token prefill chunk."""
    kwargs = {}
    if request.param is not None:
        kwargs = {
            "max_num_batched_tokens": request.param,
            "enable_chunked_prefill": True,
        }
    engine = LLM(model=MODEL_NAME, dtype="auto", gpu_memory_utilization=0.3, **kwargs)
    yield engine
    del engine
    gc.collect()
    torch.cuda.empty_cache()


def _capture(engine: LLM, prompts: list[str], max_tokens: int, pool: str | None):
    """Generate with capture at ``LAYERS`` and return the request outputs."""
    extra_args: dict = {"output_residual_stream": LAYERS}
    if pool is not None:
        extra_args["output_residual_stream_pool"] = pool
    params = SamplingParams(
        temperature=0.0, max_tokens=max_tokens, extra_args=extra_args
    )
    return engine.generate(prompts, params)


@pytest.mark.parametrize("max_tokens", [1, 5])
@pytest.mark.parametrize("pool", ["last", "mean"])
def test_pooled_capture_matches_the_full_capture(llm, pool, max_tokens):
    """The pooled row equals the same pooling of the full prompt capture."""
    prompts = [PROMPT, LONG_PROMPT]
    full_outputs = _capture(llm, prompts, max_tokens, None)
    pooled_outputs = _capture(llm, prompts, max_tokens, pool)

    for full_out, pooled_out in zip(full_outputs, pooled_outputs):
        full = full_out.activations["residual_stream"]  # type: ignore[reportAttributeAccessIssue]
        pooled = pooled_out.activations["residual_stream"]  # type: ignore[reportAttributeAccessIssue]
        n_prompt = len(full_out.prompt_token_ids)
        prompt_rows = full[:, :n_prompt].float()
        expected = (
            prompt_rows[:, -1:] if pool == "last" else prompt_rows.mean(1, keepdim=True)
        )

        assert pooled.shape == (len(LAYERS), 1, full.shape[-1])
        assert pooled.dtype == full.dtype
        torch.testing.assert_close(pooled.float(), expected, atol=1e-2, rtol=1e-2)


def test_unknown_pool_value_raises(llm):
    """A value other than ``last`` or ``mean`` is rejected before generation."""
    with pytest.raises(ValueError, match="output_residual_stream_pool"):
        _capture(llm, [PROMPT], 1, "max")


@pytest.fixture(scope="module")
async def async_engine():
    """Async engine, the path ``vllm serve`` uses."""
    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(model=MODEL_NAME, dtype="auto", gpu_memory_utilization=0.3)
    )
    yield engine
    engine.shutdown()
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("pool", ["last", "mean", "max"])
async def test_pooled_capture_on_the_async_engine(async_engine, pool):
    """The async path returns one row per layer, and rejects an unknown value."""
    params = SamplingParams(
        temperature=0.0,
        max_tokens=3,
        extra_args={
            "output_residual_stream": LAYERS,
            "output_residual_stream_pool": pool,
        },
    )
    final = None
    if pool == "max":
        with pytest.raises(ValueError, match="output_residual_stream_pool"):
            async for output in async_engine.generate(PROMPT, params, f"pool-{pool}"):
                final = output
        return
    async for output in async_engine.generate(PROMPT, params, f"pool-{pool}"):
        final = output
    assert final is not None
    pooled = final.activations["residual_stream"]  # type: ignore[reportAttributeAccessIssue]
    assert pooled.shape[:2] == (len(LAYERS), 1)

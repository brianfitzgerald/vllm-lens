"""Tests for mean-pooled capture."""

import gc

import pytest
import torch
from vllm import LLM, AsyncEngineArgs, AsyncLLMEngine, SamplingParams

from .conftest import CAPTURE_LAYERS, LONG_PROMPT, MODEL_NAME, PROMPT

# A request option and the rows of the full capture that it must equal.
MODES = {
    "mean": (
        {"output_residual_stream_pool": "mean"},
        lambda rows, n_prompt: rows[:, :n_prompt].mean(1, keepdim=True),
    ),
}


def _capture(engine: LLM, prompts: list[str], max_tokens: int, extra: dict):
    """Generate with capture at ``CAPTURE_LAYERS`` and return the request outputs."""
    params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        extra_args={"output_residual_stream": CAPTURE_LAYERS, **extra},
    )
    return engine.generate(prompts, params)


@pytest.mark.parametrize("max_tokens", [1, 5])
@pytest.mark.parametrize("mode", MODES)
def test_capture_matches_the_full_capture(capture_llm, mode, max_tokens):
    """The pooled rows equal the same rows of the full capture."""
    extra, expected_rows = MODES[mode]
    prompts = [PROMPT, LONG_PROMPT]
    full_outputs = _capture(capture_llm, prompts, max_tokens, {})
    outputs = _capture(capture_llm, prompts, max_tokens, extra)

    for full_out, out in zip(full_outputs, outputs):
        full = full_out.activations["residual_stream"]  # type: ignore[reportAttributeAccessIssue]
        got = out.activations["residual_stream"]  # type: ignore[reportAttributeAccessIssue]
        n_prompt = len(full_out.prompt_token_ids)
        expected = expected_rows(full.float(), n_prompt)
        assert got.shape == expected.shape and got.dtype == full.dtype
        torch.testing.assert_close(got.float(), expected, atol=1e-2, rtol=1e-2)


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


async def _generate_async(engine: AsyncLLMEngine, request_id: str, extra: dict):
    """Run ``PROMPT`` with capture at ``CAPTURE_LAYERS`` and return the final output."""
    params = SamplingParams(
        temperature=0.0,
        max_tokens=3,
        extra_args={"output_residual_stream": CAPTURE_LAYERS, **extra},
    )
    final = None
    async for output in engine.generate(PROMPT, params, request_id):
        final = output
    assert final is not None
    return final


@pytest.mark.parametrize("mode", MODES)
async def test_capture_on_the_async_engine(async_engine, mode):
    """The async path returns the same rows of its own full capture."""
    extra, expected_rows = MODES[mode]
    full_out = await _generate_async(async_engine, f"full-{mode}", {})
    out = await _generate_async(async_engine, f"mode-{mode}", extra)
    full = full_out.activations["residual_stream"].float()  # type: ignore[reportAttributeAccessIssue]
    expected = expected_rows(full, len(full_out.prompt_token_ids))
    torch.testing.assert_close(
        out.activations["residual_stream"].float(),  # type: ignore[reportAttributeAccessIssue]
        expected,
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize(
    "extra",
    [
        {"output_residual_stream_pool": "last"},
        {"output_residual_stream": None, "output_residual_stream_pool": "mean"},
    ],
    ids=["last", "no-capture"],
)
async def test_bad_capture_options_raise_on_the_async_engine(
    async_engine, extra, request
):
    """The async path rejects a bad pool value or a pool without capture."""
    with pytest.raises(ValueError, match="output_residual_stream"):
        await _generate_async(async_engine, request.node.callspec.id, extra)

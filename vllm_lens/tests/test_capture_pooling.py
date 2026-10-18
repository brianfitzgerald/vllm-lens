"""Tests for mean-pooled and position-selected capture."""

import gc

import pytest
import torch
from vllm import LLM, AsyncEngineArgs, AsyncLLMEngine, SamplingParams

from vllm_lens._helpers._positions import POSITIONS_KEY, POSITIONS_META_KEY
from vllm_lens._worker_ext import HiddenStatesExtension, _capture_rows

from .conftest import (
    CAPTURE_LAYERS,
    LAYER_IDX,
    LONG_PROMPT,
    MODEL_NAME,
    PROMPT,
    fake_extension,
)

# A request option and the rows of the full capture that it must equal.
MODES = {
    "mean": (
        {"output_residual_stream_pool": "mean"},
        lambda rows, n_prompt: rows[:, :n_prompt].mean(1, keepdim=True),
    ),
    "last-prompt-token": (
        {POSITIONS_KEY: "last_prompt_token"},
        lambda rows, n_prompt: rows[:, n_prompt - 1 : n_prompt],
    ),
    "positions": (
        {POSITIONS_KEY: [0, 3, 1000]},
        lambda rows, n_prompt: rows[:, [0, 3]],
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
    """The pooled or selected rows equal the same rows of the full capture."""
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
        if POSITIONS_KEY in extra:
            positions = out.activations[POSITIONS_META_KEY]  # type: ignore[reportAttributeAccessIssue]
            assert positions == (
                [n_prompt - 1] if mode == "last-prompt-token" else [0, 3]
            )


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
        {"output_residual_stream_pool": "mean", POSITIONS_KEY: [0]},
        {POSITIONS_KEY: [-1]},
    ],
    ids=["last", "pool-and-positions", "negative"],
)
async def test_bad_capture_options_raise_on_the_async_engine(async_engine, extra):
    """The async path rejects a bad pool value, a combination, or a bad selector."""
    with pytest.raises(ValueError, match="output_residual_stream"):
        await _generate_async(async_engine, f"bad-{len(extra)}", extra)


# Forward steps as (first absolute position, number of rows). The prompt has
# 10 positions; positions 10 and 11 are decode steps.
STEP_SCHEDULES = {
    "single-pass": [(0, 10), (10, 1), (11, 1)],
    "chunked": [(0, 4), (4, 4), (8, 2), (10, 1)],
    # A preemption after two chunks: the prefill starts again at position 0.
    "preempted": [(0, 4), (4, 4), (0, 4), (4, 4), (8, 2), (10, 1)],
    # The last chunk also holds the first decode position.
    "chunk-past-the-prompt": [(0, 6), (6, 5)],
}


def _step(extension, rows: torch.Tensor, first: int) -> None:
    """Capture ``rows`` as one forward step of ``r-0`` that starts at position ``first``."""
    extension.model_runner.requests["r-0"].num_computed_tokens = first
    _capture_rows(extension, LAYER_IDX, rows, torch.tensor([0, rows.shape[0]]))


@pytest.mark.parametrize("schedule", STEP_SCHEDULES)
@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ({"output_residual_stream_pool": "mean"}, lambda s: s[:10].mean(0, True)),
        ({POSITIONS_KEY: "last_prompt_token"}, lambda s: s[9:10]),
        ({POSITIONS_KEY: [1, 5, 10]}, lambda s: s[[1, 5, 10]]),
    ],
    ids=["mean", "last-prompt-token", "positions"],
)
def test_each_position_is_captured_once(schedule, extra, expected):
    """The capture is exact for any step schedule, with a re-prefill too. No GPU."""
    stream = torch.randn(12, 8, generator=torch.Generator().manual_seed(0)).to(
        torch.bfloat16
    )
    extension = fake_extension(
        {"r-0": {"output_residual_stream": [LAYER_IDX], **extra}}
    )
    for first, n_rows in STEP_SCHEDULES[schedule]:
        _step(extension, stream[first : first + n_rows], first)

    activations = HiddenStatesExtension._build_payload(extension, "r-0")["activations"]
    got = activations["residual_stream"][0]
    want = expected(stream.float())
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got.float(), want, atol=1e-2, rtol=1e-2)


def test_a_decode_step_adds_nothing_to_the_mean() -> None:
    """A decode step stores no pooled entry. No GPU."""
    extension = fake_extension(
        {
            "r-0": {
                "output_residual_stream": [LAYER_IDX],
                "output_residual_stream_pool": "mean",
            }
        }
    )
    _step(extension, torch.zeros(1, 8), 10)
    assert extension._captured_states == {} and extension._pooled == {}

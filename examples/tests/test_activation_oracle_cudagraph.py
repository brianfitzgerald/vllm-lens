"""Runs the activation oracle example in eager mode and under CUDA graphs."""

import gc

import pandas as pd
import pytest
import torch
from vllm import AsyncEngineArgs, AsyncLLMEngine

from activation_oracle import (
    LAYER,
    LORA_PATH,
    MODEL_NAME,
    TARGET_MESSAGES,
    collect_activations,
    get_target_prompt_input_ids,
    load_ao_config,
    run_oracle_sweep,
)

NAMES = ["Socrates", "Plato", "Aristotle"]
MODES = ["eager", "cudagraph"]


@pytest.fixture(scope="module")
async def sweeps() -> dict[str, pd.DataFrame]:
    """The oracle sweep of each mode; in eager mode, `unsteered` is the same sweep with
    scale 0, which shows that steering is what names the philosophers."""
    ao_config = load_ao_config(LORA_PATH)
    lens = {"capture_layers": [LAYER], "steer_layers": [ao_config.hook_onto_layer]}
    frames: dict[str, pd.DataFrame] = {}
    for mode in MODES:
        engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(
                model=MODEL_NAME,
                enable_lora=True,
                max_lora_rank=64,
                max_model_len=4096,
                gpu_memory_utilization=0.4,
                additional_config={"vllm_lens": lens} if mode == "cudagraph" else {},
            )
        )
        assert engine.vllm_config.model_config.enforce_eager == (mode == "eager")
        tokenizer = engine.tokenizer
        prompt_ids = get_target_prompt_input_ids(TARGET_MESSAGES, tokenizer)  # type: ignore[arg-type]
        residual = await collect_activations(engine, prompt_ids, LAYER)
        frame = await run_oracle_sweep(
            engine,
            tokenizer,  # type: ignore[arg-type]
            ao_config,
            prompt_ids,
            residual,
        )
        if mode == "eager":
            unsteered = await run_oracle_sweep(
                engine,
                tokenizer,  # type: ignore[arg-type]
                ao_config.model_copy(update={"steering_coefficient": 0.0}),
                prompt_ids,
                residual,
            )
            frame["unsteered"] = unsteered["oracle_response"]
        frames[mode] = frame
        engine.shutdown()
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    return frames


@pytest.mark.parametrize("mode", MODES)
def test_oracle_names_the_philosophers(
    sweeps: dict[str, pd.DataFrame], mode: str
) -> None:
    """Each philosopher is named at some position, and, in eager mode, at none with scale 0."""
    frame = sweeps[mode]
    for name in NAMES:
        assert frame["oracle_response"].str.contains(name).any(), frame
        if "unsteered" in frame:
            assert not frame["unsteered"].str.contains(name).any(), frame


def test_cudagraph_responses_match_eager(sweeps: dict[str, pd.DataFrame]) -> None:
    """The kernels differ between the modes: 25 of 26 greedy responses were the same."""
    eager = sweeps["eager"]["oracle_response"]
    graph = sweeps["cudagraph"]["oracle_response"]
    assert (eager == graph).mean() >= 0.8, pd.DataFrame(
        {"eager": eager, "cudagraph": graph}
    )

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
    """The oracle sweep of each mode; `unsteered` is the same sweep with scale 0."""
    ao_config = load_ao_config(LORA_PATH)
    env = {
        "VLLM_LENS_CUDAGRAPH": "1",
        "VLLM_LENS_CAPTURE_LAYERS": str(LAYER),
        "VLLM_LENS_STEER_LAYERS": str(ao_config.hook_onto_layer),
    }
    frames: dict[str, pd.DataFrame] = {}
    for mode in MODES:
        with pytest.MonkeyPatch.context() as patch:
            for name, value in (env if mode == "cudagraph" else {}).items():
                patch.setenv(name, value)
            engine = AsyncLLMEngine.from_engine_args(
                AsyncEngineArgs(
                    model=MODEL_NAME,
                    enable_lora=True,
                    max_lora_rank=64,
                    max_model_len=4096,
                    gpu_memory_utilization=0.4,
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
    """Each philosopher is named at some position, and at none with scale 0."""
    frame = sweeps[mode]
    for name in NAMES:
        assert frame["oracle_response"].str.contains(name).any(), frame
        assert not frame["unsteered"].str.contains(name).any(), frame


def test_cudagraph_responses_match_eager(sweeps: dict[str, pd.DataFrame]) -> None:
    """The kernels differ between the modes, so a few greedy responses can differ."""
    eager = sweeps["eager"]["oracle_response"]
    graph = sweeps["cudagraph"]["oracle_response"]
    assert (eager == graph).mean() >= 0.8, pd.DataFrame(
        {"eager": eager, "cudagraph": graph}
    )

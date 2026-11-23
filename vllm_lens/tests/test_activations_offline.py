import gc
import os
import subprocess
import sys

import pytest
import torch
from vllm import LLM, SamplingParams

from .conftest import (
    ARMED,
    LAYER_IDX,
    MODEL_NAME,
    NUM_LAYERS,
    PROMPT,
    PROMPTS,
    REFERENCE_DTYPE,
)

# Two layers far apart, so a request's rows of both layers are checked.
_CROSS_STAGE_LAYERS = [LAYER_IDX, NUM_LAYERS - 4]


@pytest.fixture(scope="module")
def llm_reference():
    """FP32 engine for HF parity; llm_model retains native-dtype coverage."""
    llm = LLM(
        model=MODEL_NAME,
        additional_config=ARMED,
        dtype=REFERENCE_DTYPE,
        gpu_memory_utilization=0.3,
    )
    yield llm
    del llm
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def llm_model():
    llm = LLM(
        model=MODEL_NAME,
        additional_config=ARMED,
        dtype="auto",
        gpu_memory_utilization=0.3,
    )
    yield llm
    del llm
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def llm_model_tp2():
    """Tensor-parallel (TP=2) offline engine for batched-fetch coverage."""
    llm = LLM(
        model=MODEL_NAME,
        additional_config=ARMED,
        dtype="auto",
        gpu_memory_utilization=0.3,
        tensor_parallel_size=2,
    )
    yield llm
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def _get_hf_acts(model, tokenizer, prompt: str) -> torch.Tensor:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    num_tokens = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    return out.hidden_states[LAYER_IDX + 1][0, :num_tokens].float()


def _get_vllm_acts(llm: LLM, prompts: list[str], max_tokens: int) -> list[torch.Tensor]:
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        extra_args={"output_residual_stream": [LAYER_IDX]},
    )
    outputs = llm.generate(prompts, sampling_params)
    return [out.activations["residual_stream"] for out in outputs]  # type: ignore[reportAttributeAccessIssue]


def _assert_close(vllm_acts: torch.Tensor, hf_acts: torch.Tensor):
    assert vllm_acts.shape == hf_acts.shape, (
        f"Shape mismatch: vLLM {vllm_acts.shape} vs HF {hf_acts.shape}"
    )
    mean_abs_diff = (vllm_acts - hf_acts.to(vllm_acts.device)).abs().mean().item()
    assert mean_abs_diff < 1e-2, f"Mean abs diff too large: {mean_abs_diff:.6f}"


class TestOfflineMatchesTransformers:
    def test_single_prompt_1_token(self, llm_reference, hf_model):
        model, tokenizer = hf_model
        num_tokens = tokenizer(PROMPT, return_tensors="pt").input_ids.shape[1]

        hf_acts = _get_hf_acts(model, tokenizer, PROMPT)
        streams = _get_vllm_acts(llm_reference, [PROMPT], max_tokens=1)
        vllm_acts = streams[0][0, :num_tokens].float()

        _assert_close(vllm_acts, hf_acts)

    def test_single_prompt_10_tokens(self, llm_reference, hf_model):
        model, tokenizer = hf_model
        num_tokens = tokenizer(PROMPT, return_tensors="pt").input_ids.shape[1]

        hf_acts = _get_hf_acts(model, tokenizer, PROMPT)
        streams = _get_vllm_acts(llm_reference, [PROMPT], max_tokens=10)
        vllm_acts = streams[0][0, :num_tokens].float()

        _assert_close(vllm_acts, hf_acts)

    def test_batch_prompts_10_tokens(self, llm_reference, hf_model):
        model, tokenizer = hf_model

        streams = _get_vllm_acts(llm_reference, PROMPTS, max_tokens=10)

        for i, prompt in enumerate(PROMPTS):
            num_tokens = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
            hf_acts = _get_hf_acts(model, tokenizer, prompt)
            vllm_acts = streams[i][0, :num_tokens].float()
            _assert_close(vllm_acts, hf_acts)

    def test_activation_shape_matches_token_count(self, llm_model):
        """Activation seq dim must equal prompt + generated - 1."""
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=10,
            extra_args={"output_residual_stream": [LAYER_IDX]},
        )
        outputs = llm_model.generate(PROMPTS, sampling_params)
        for output in outputs:
            n_prompt = len(output.prompt_token_ids)
            n_gen = len(output.outputs[0].token_ids)
            expected = n_prompt + n_gen - 1
            rs = output.activations["residual_stream"]
            assert rs.shape[1] == expected, (
                f"Expected seq_len={expected} (prompt={n_prompt}, gen={n_gen}), "
                f"got {rs.shape[1]}"
            )


def _capture_batch(
    llm: LLM, prompts: list[str], layers: list[int]
) -> list[torch.Tensor]:
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        extra_args={"output_residual_stream": layers},
    )
    return [out.activations["residual_stream"] for out in llm.generate(prompts, sp)]  # type: ignore[reportAttributeAccessIssue]


def _assert_batched_matches_individual(llm: LLM, layers: list[int]) -> None:
    """Batched capture of many prompts must equal capturing each alone.

    Isolates the batched-fetch demux/merge from parallel-vs-HF numerics:
    both paths run the same forward on the same engine, so any difference is
    a request-routing or cross-rank-merge error, not TP/PP fp drift.
    """
    batched = _capture_batch(llm, PROMPTS, layers)
    assert len(batched) == len(PROMPTS)
    for i, prompt in enumerate(PROMPTS):
        single = _capture_batch(llm, [prompt], layers)[0]
        assert batched[i].shape == single.shape, (
            f"prompt {i}: batched shape {batched[i].shape} != single "
            f"{single.shape} — request demux is misrouted"
        )
        assert batched[i].shape[0] == len(layers), (
            f"prompt {i}: expected {len(layers)} captured layers, got "
            f"{batched[i].shape[0]} — cross-rank merge dropped/duplicated layers"
        )
        # Scale-relative tolerance: batched vs single differ only by batched-GEMM
        # fp jitter (≈1%), whereas a misrouted request would differ by ~100%.
        b, s = batched[i].float(), single.float()
        rel = (b - s).abs().mean().item() / (s.abs().mean().item() + 1e-6)
        assert rel < 0.05, (
            f"prompt {i}: batched vs single relative mean abs diff {rel:.4f} "
            f"too large — wrong request's activations returned"
        )


class TestBatchedFetch:
    """Each request of one ``LLM.generate`` call gets its own capture file, also under
    tensor parallelism, where rank 0 alone captures."""

    def test_batched_fetch_matches_individual(self, llm_model):
        _assert_batched_matches_individual(llm_model, [LAYER_IDX])

    @pytest.mark.skipif(
        torch.cuda.device_count() < 2, reason="TP=2 requires at least 2 GPUs"
    )
    def test_batched_fetch_matches_individual_tp2(self, llm_model_tp2):
        _assert_batched_matches_individual(llm_model_tp2, _CROSS_STAGE_LAYERS)


def test_in_process_engine_seals_last_captures():
    """An in-process engine (no engine-core process) returns the captures of its last batch."""
    script = f"""
from vllm import LLM, SamplingParams
llm = LLM(model={MODEL_NAME!r}, additional_config={ARMED!r}, gpu_memory_utilization=0.2)
sp = SamplingParams(max_tokens=4, extra_args={{"output_residual_stream": [{LAYER_IDX}]}})
outputs = llm.generate({PROMPTS[:2]!r}, sp)
for out in outputs:
    rows = len(out.prompt_token_ids) + len(out.outputs[0].token_ids) - 1
    assert out.activations["residual_stream"].shape[:2] == (1, rows)
"""
    env = {
        **os.environ,
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_LENS_CAPTURE_TIMEOUT": "30",
    }
    subprocess.run([sys.executable, "-c", script], check=True, env=env, timeout=900)


def test_offline_captures_larger_than_the_arena():
    """An offline call whose captures need more than the arena still gets every capture:
    the call's collector releases each one as the worker seals it. Four requests run at
    once, so the running captures fit."""
    from .conftest import lens_engine

    lens = {**ARMED["vllm_lens"], "capture_arena_gib": 0.25}
    engines = lens_engine(lens, enable_prefix_caching=False, max_num_seqs=4)
    llm = next(engines)
    try:
        prompts = [
            f"{i} " + "The river runs past the old mill. " * 50 for i in range(16)
        ]
        sp = SamplingParams(
            max_tokens=8,
            ignore_eos=True,
            extra_args={"output_residual_stream": list(range(NUM_LAYERS))},
        )
        outputs = llm.generate(prompts, sp)
        total = 0
        for out in outputs:
            rows = len(out.prompt_token_ids) + len(out.outputs[0].token_ids) - 1
            assert out.activations["residual_stream"].shape[:2] == (NUM_LAYERS, rows)
            total += (
                rows * NUM_LAYERS * llm.llm_engine.model_config.get_hidden_size() * 2
            )
        assert total > 0.25 * (1 << 30)
    finally:
        engines.close()

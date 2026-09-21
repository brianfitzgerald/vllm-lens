"""Tests for steering under CUDA graphs via ``VLLM_LENS_STEER_LAYERS``."""

import gc

import pytest
import torch
from vllm import LLM, SamplingParams

from vllm_lens import SteeringVector

from .conftest import make_llm, prompt_rows, row_error

LONG_PROMPT = "The quick brown fox jumps over the lazy dog. " * 20
SHORT_PROMPTS = ["The future of AI is", "Paris is the capital of", "Water boils at"]
STEER_LAYER = 2


@pytest.fixture(scope="module")
def graph_llm():
    """CUDA graphs on, 64-token prefill chunks, steering and capture on one layer."""
    env = {
        "VLLM_LENS_STEER_LAYERS": str(STEER_LAYER),
        "VLLM_LENS_CAPTURE_LAYERS": str(STEER_LAYER),
    }
    # Prefix caching off: a later request must not read the KV of a steered one.
    engine = make_llm(
        env,
        max_num_batched_tokens=64,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
    )
    yield engine
    del engine
    gc.collect()
    torch.cuda.empty_cache()


def _vector(seed: int, hidden: int, **kwargs) -> SteeringVector:
    """A reproducible random vector on ``STEER_LAYER``; 3D when positions are given."""
    generator = torch.Generator().manual_seed(seed)
    shape = (1, hidden)
    if "position_indices" in kwargs:
        shape = (1, len(kwargs["position_indices"]), hidden)
    return SteeringVector(
        activations=torch.randn(shape, generator=generator),
        layer_indices=[STEER_LAYER],
        **kwargs,
    )


def _run(
    engine: LLM,
    vectors: list[list[SteeringVector]],
    max_tokens: int = 4,
    prompts: list[str] | None = None,
):
    """Generate one prompt per entry of ``vectors``, in one batch, with capture."""
    params = []
    for entry in vectors:
        extra_args: dict = {"output_residual_stream": [STEER_LAYER]}
        if entry:
            extra_args["apply_steering_vectors"] = entry
        params.append(
            SamplingParams(
                temperature=0.0, max_tokens=max_tokens, extra_args=extra_args
            )
        )
    return engine.generate(prompts or [LONG_PROMPT] * len(vectors), params)


VECTOR_CASES = {
    "add": lambda hidden: [_vector(0, hidden, scale=8.0)],
    "norm-match": lambda hidden: [_vector(1, hidden, scale=4.0, norm_match=True)],
    "two-vectors-one-layer": lambda hidden: [
        _vector(2, hidden, scale=8.0),
        _vector(3, hidden, scale=2.0, norm_match=True),
    ],
    "prompt-position": lambda hidden: [
        _vector(4, hidden, scale=4.0, norm_match=True, position_indices=[0, 100])
    ],
}


@pytest.mark.parametrize("case", VECTOR_CASES)
def test_buffer_steering_matches_eager_steering(eager_llm, graph_llm, hidden, case):
    """Every steered row under CUDA graphs equals the eager forward-hook row."""
    eager = _run(eager_llm, [VECTOR_CASES[case](hidden)])[0]
    graph = _run(graph_llm, [VECTOR_CASES[case](hidden)])[0]
    error = row_error(prompt_rows(graph), prompt_rows(eager))
    assert error < 5e-2, f"largest row error {error:.4f}"


def test_several_requests_share_one_forward_pass(eager_llm, graph_llm, hidden):
    """Short prompts prefill in one step: two steered requests and one unsteered."""
    vectors = [
        [_vector(8, hidden, scale=8.0)],
        [],
        [_vector(9, hidden, scale=4.0, norm_match=True, position_indices=[1])],
    ]
    eager = _run(eager_llm, vectors, prompts=SHORT_PROMPTS)
    graph = _run(graph_llm, vectors, prompts=SHORT_PROMPTS)
    unsteered = _run(graph_llm, [[], [], []], prompts=SHORT_PROMPTS)
    for index, (eager_out, graph_out) in enumerate(zip(eager, graph)):
        error = row_error(prompt_rows(graph_out), prompt_rows(eager_out))
        assert error < 5e-2, f"request {index}: largest row error {error:.4f}"
    # The unsteered request is not changed by its two neighbours.
    assert row_error(prompt_rows(graph[1]), prompt_rows(unsteered[1])) < 5e-2
    assert row_error(prompt_rows(graph[0]), prompt_rows(unsteered[0])) > 0.1


def test_a_decode_position_is_steered_alone(graph_llm, hidden):
    """A position in the generated part is steered in its decode step, and only there."""
    n_prompt = len(_run(graph_llm, [[]], max_tokens=1)[0].prompt_token_ids)
    position = n_prompt + 2
    vector = _vector(5, hidden, scale=4.0, norm_match=True, position_indices=[position])
    steered = _run(graph_llm, [[vector]], max_tokens=8)[0]
    rows = steered.activations["residual_stream"][0].float()  # type: ignore[reportAttributeAccessIssue]
    direction = vector.activations[0, 0].float()
    cosines = torch.nn.functional.cosine_similarity(rows, direction.unsqueeze(0))
    # A steered row has cosine ~0.97 with the vector; other rows have ~0.
    assert torch.nonzero(cosines > 0.5).flatten().tolist() == [position]


def test_scale_zero_matches_the_baseline(graph_llm, hidden):
    """A vector with scale 0 changes no token."""
    baseline, steered = _run(graph_llm, [[], [_vector(6, hidden, scale=0.0)]])
    assert steered.outputs[0].token_ids == baseline.outputs[0].token_ids


def test_steering_does_not_leak_to_later_requests(graph_llm, hidden):
    """An unsteered request is the same before and after a steered one."""
    before = _run(graph_llm, [[]])[0]
    steered = _run(graph_llm, [[_vector(7, hidden, scale=8.0)]])[0]
    after = _run(graph_llm, [[]])[0]
    assert row_error(prompt_rows(steered), prompt_rows(before)) > 0.1
    assert after.outputs[0].token_ids == before.outputs[0].token_ids
    assert row_error(prompt_rows(after), prompt_rows(before)) < 5e-2


def test_a_bad_vector_does_not_stop_the_engine(graph_llm, hidden):
    """A vector of the wrong width is logged and skipped, as in eager mode."""
    bad = SteeringVector(
        activations=torch.ones(1, hidden + 1), layer_indices=[STEER_LAYER]
    )
    baseline, skipped = _run(graph_llm, [[], [bad]])
    assert skipped.outputs[0].token_ids == baseline.outputs[0].token_ids
    steered = _run(graph_llm, [[_vector(10, hidden, scale=8.0)]])[0]
    assert row_error(prompt_rows(steered), prompt_rows(baseline)) > 0.1


def test_a_layer_outside_the_list_is_rejected(graph_llm, hidden):
    """A vector on a layer outside ``VLLM_LENS_STEER_LAYERS`` raises before generation."""
    vector = SteeringVector(
        activations=torch.zeros(1, hidden), layer_indices=[STEER_LAYER + 1]
    )
    with pytest.raises(ValueError, match="VLLM_LENS_STEER_LAYERS"):
        _run(graph_llm, [[vector]])

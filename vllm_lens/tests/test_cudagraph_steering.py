"""Tests for steering under CUDA graphs via ``VLLM_LENS_STEER_LAYERS``."""

import gc

import pytest
import torch
from vllm import LLM, SamplingParams

from vllm_lens import SteeringVector

from .conftest import MODEL_NAME

LONG_PROMPT = "The quick brown fox jumps over the lazy dog. " * 20
STEER_LAYER = 2
HIDDEN = 896


def _make_llm(env: dict[str, str], **kwargs):
    """Start an offline engine with ``env`` set, so its workers inherit it."""
    with pytest.MonkeyPatch.context() as patch:
        for name, value in env.items():
            patch.setenv(name, value)
        # Prefix caching off: a steered request writes steered KV, and a later
        # request that read it would look like a leak from the buffers.
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
    """CUDA graphs on, 64-token prefill chunks, steering and capture on one layer."""
    env = {
        "VLLM_LENS_STEER_LAYERS": str(STEER_LAYER),
        "VLLM_LENS_CAPTURE_LAYERS": str(STEER_LAYER),
    }
    engine = _make_llm(env, max_num_batched_tokens=64, enable_chunked_prefill=True)
    yield engine
    del engine
    gc.collect()
    torch.cuda.empty_cache()


def _vector(seed: int, **kwargs) -> SteeringVector:
    """A reproducible random vector on ``STEER_LAYER``; 3D when positions are given."""
    generator = torch.Generator().manual_seed(seed)
    shape = (1, HIDDEN)
    if "position_indices" in kwargs:
        shape = (1, len(kwargs["position_indices"]), HIDDEN)
    return SteeringVector(
        activations=torch.randn(shape, generator=generator),
        layer_indices=[STEER_LAYER],
        **kwargs,
    )


def _run(engine: LLM, vectors: list[list[SteeringVector]], max_tokens: int = 4):
    """Generate ``LONG_PROMPT`` once per entry of ``vectors``, in one batch, with capture."""
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
    return engine.generate([LONG_PROMPT] * len(vectors), params)


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """``||actual - expected|| / ||expected||`` in float32."""
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


VECTOR_CASES = {
    "add": lambda: [_vector(0, scale=8.0)],
    "norm-match": lambda: [_vector(1, scale=4.0, norm_match=True)],
    "two-vectors-one-layer": lambda: [
        _vector(2, scale=8.0),
        _vector(3, scale=2.0, norm_match=True),
    ],
    "prompt-position": lambda: [
        _vector(4, scale=4.0, norm_match=True, position_indices=[0, 100])
    ],
}


@pytest.mark.parametrize("case", VECTOR_CASES)
def test_buffer_steering_matches_eager_steering(eager_llm, graph_llm, case):
    """The steered stream under CUDA graphs equals the eager forward-hook result."""
    eager = _run(eager_llm, [VECTOR_CASES[case]()])[0]
    graph = _run(graph_llm, [VECTOR_CASES[case]()])[0]
    eager_acts = eager.activations["residual_stream"]  # type: ignore[reportAttributeAccessIssue]
    graph_acts = graph.activations["residual_stream"]  # type: ignore[reportAttributeAccessIssue]
    n_prompt = len(eager.prompt_token_ids)
    # The prompt rows do not depend on the sampled tokens.
    error = _relative_error(graph_acts[:, :n_prompt], eager_acts[:, :n_prompt])
    assert error < 2e-2, f"relative error {error:.4f}"


def test_a_decode_position_is_steered_alone(graph_llm):
    """A position in the generated part is steered in its decode step, and only there."""
    n_prompt = len(_run(graph_llm, [[]], max_tokens=1)[0].prompt_token_ids)
    position = n_prompt + 2
    vector = _vector(5, scale=4.0, norm_match=True, position_indices=[position])
    steered = _run(graph_llm, [[vector]], max_tokens=8)[0]
    rows = steered.activations["residual_stream"][0].float()  # type: ignore[reportAttributeAccessIssue]
    direction = vector.activations[0, 0].float()
    cosines = torch.nn.functional.cosine_similarity(rows, direction.unsqueeze(0))
    # A steered row has cosine ~0.97 with the vector; other rows have ~0.
    assert torch.nonzero(cosines > 0.5).flatten().tolist() == [position]


def test_scale_zero_matches_the_baseline(graph_llm):
    """A vector with scale 0 changes no token."""
    baseline, steered = _run(graph_llm, [[], [_vector(6, scale=0.0)]])
    assert steered.outputs[0].token_ids == baseline.outputs[0].token_ids


def test_steering_does_not_leak_to_other_requests(graph_llm):
    """An unsteered request is the same in a batch with a steered one, and after it."""
    alone = _run(graph_llm, [[]])[0]
    beside, steered = _run(graph_llm, [[], [_vector(7, scale=8.0)]])
    after = _run(graph_llm, [[]])[0]
    alone_acts = alone.activations["residual_stream"]  # type: ignore[reportAttributeAccessIssue]
    steered_acts = steered.activations["residual_stream"]  # type: ignore[reportAttributeAccessIssue]
    assert _relative_error(steered_acts, alone_acts) > 0.1
    for other in (beside, after):
        other_acts = other.activations["residual_stream"]  # type: ignore[reportAttributeAccessIssue]
        assert other.outputs[0].token_ids == alone.outputs[0].token_ids
        assert _relative_error(other_acts, alone_acts) < 2e-2


def test_an_unarmed_layer_is_rejected(graph_llm):
    """A vector on a layer outside ``VLLM_LENS_STEER_LAYERS`` raises before generation."""
    vector = SteeringVector(
        activations=torch.zeros(1, HIDDEN), layer_indices=[STEER_LAYER + 1]
    )
    with pytest.raises(ValueError, match="VLLM_LENS_STEER_LAYERS"):
        _run(graph_llm, [[vector]])

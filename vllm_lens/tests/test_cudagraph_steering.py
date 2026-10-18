"""Tests for steering under CUDA graphs via ``additional_config["vllm_lens"]``."""

import gc

import pytest
import torch
from vllm import LLM, SamplingParams

from vllm_lens import SteeringVector

from .conftest import make_llm

LONG_PROMPT = "The quick brown fox jumps over the lazy dog. " * 20
SHORT_PROMPTS = ["The future of AI is", "Paris is the capital of", "Water boils at"]
STEER_LAYER = 2


def row_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """The largest per-row ``||actual - expected|| / ||expected||``."""
    return ((actual - expected).norm(dim=-1) / expected.norm(dim=-1)).max().item()


def prompt_rows(output) -> torch.Tensor:
    """The captured prompt rows of the first layer, which no sampled token changes."""
    acts = output.activations["residual_stream"]  # type: ignore[reportAttributeAccessIssue]
    return acts[0, : len(output.prompt_token_ids)].float()


@pytest.fixture(scope="module")
def hidden(eager_llm) -> int:
    """The hidden size of the test model."""
    return eager_llm.llm_engine.vllm_config.model_config.get_hidden_size()


@pytest.fixture(scope="module")
def graph_llm():
    """CUDA graphs on, 64-token prefill chunks, steering and capture on one layer."""
    env = {"steer_layers": [STEER_LAYER], "capture_layers": [STEER_LAYER]}
    engine = make_llm(env, max_num_batched_tokens=64, enable_chunked_prefill=True)
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


@pytest.fixture(scope="module")
def op_eager_llm():
    """The steer and capture ops with ``enforce_eager``: the kernels of ``eager_llm``."""
    env = {"steer_layers": [STEER_LAYER], "capture_layers": [STEER_LAYER]}
    engine = make_llm(env, enforce_eager=True, enable_prefix_caching=False)
    yield engine
    del engine
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("case", VECTOR_CASES)
def test_op_steering_matches_the_forward_hooks(eager_llm, op_eager_llm, hidden, case):
    """With the same kernels, the steer op gives the rows the forward hooks give."""
    eager = _run(eager_llm, [VECTOR_CASES[case](hidden)])[0]
    op = _run(op_eager_llm, [VECTOR_CASES[case](hidden)])[0]
    error = row_error(prompt_rows(op), prompt_rows(eager))
    # Both paths run the same float32 steering math on the same kernels.
    assert error < 1e-3, f"largest row error {error:.4f}"


@pytest.mark.parametrize("case", VECTOR_CASES)
def test_graph_steering_matches_eager_steering(eager_llm, graph_llm, hidden, case):
    """Compiled kernels differ from eager ones, so the limit is 5% per row."""
    eager = _run(eager_llm, [VECTOR_CASES[case](hidden)])[0]
    graph = _run(graph_llm, [VECTOR_CASES[case](hidden)])[0]
    error = row_error(prompt_rows(graph), prompt_rows(eager))
    assert error < 5e-2, f"largest row error {error:.4f}"


def test_interleaved_requests_match_eager(eager_llm, graph_llm, hidden):
    """Six prompts in one step: two 2D vectors, a norm-match vector, a positional vector
    and two unsteered requests, each row equal to eager's."""
    vectors = [
        [_vector(11, hidden, scale=8.0)],
        [],
        [_vector(12, hidden, scale=4.0, norm_match=True)],
        [_vector(13, hidden, scale=6.0)],
        [_vector(14, hidden, scale=4.0, norm_match=True, position_indices=[1])],
        [],
    ]
    prompts = SHORT_PROMPTS * 2
    eager = _run(eager_llm, vectors, prompts=prompts)
    graph = _run(graph_llm, vectors, prompts=prompts)
    unsteered = _run(graph_llm, [[]] * len(prompts), prompts=prompts)
    for index, (eager_out, graph_out) in enumerate(zip(eager, graph)):
        error = row_error(prompt_rows(graph_out), prompt_rows(eager_out))
        assert error < 5e-2, f"request {index}: largest row error {error:.4f}"
    # The unsteered requests are not changed by their neighbours.
    for index in (1, 5):
        assert (
            row_error(prompt_rows(graph[index]), prompt_rows(unsteered[index])) < 5e-2
        )
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
    """A vector on a layer outside ``steer_layers`` raises before generation."""
    vector = SteeringVector(
        activations=torch.zeros(1, hidden), layer_indices=[STEER_LAYER + 1]
    )
    with pytest.raises(ValueError, match="steer_layers"):
        _run(graph_llm, [[vector]])


@pytest.mark.parametrize("fused", [False, True], ids=["plain", "fused-residual"])
def test_the_steer_op_steers_then_the_capture_op_copies(fused):
    """The steer op adds each token's row, padding rows take row 0, and the capture op then
    copies the steered stream. No GPU."""
    import vllm_lens._graph_worker  # noqa: F401
    from vllm_lens._worker_ext import _steer

    generator = torch.Generator().manual_seed(0)
    hidden_states = torch.randn(4, 8, generator=generator)
    residual = torch.randn(4, 8, generator=generator) if fused else None
    add = torch.randn(3, 8, generator=generator)
    norm = torch.randn(3, 8, generator=generator)
    add[0] = norm[0] = 0
    rows = torch.tensor([1, 0, 2, 1, 2, 2], dtype=torch.int32)
    expected = hidden_states.clone()
    _steer(expected, residual, add, norm, rows[:4])
    out = torch.zeros(6, 8)
    torch.ops.vllm_lens.steer(hidden_states, residual, add, norm, rows)
    torch.ops.vllm_lens.capture(hidden_states, residual, out)
    torch.testing.assert_close(hidden_states, expected)
    stream = expected if residual is None else expected + residual
    torch.testing.assert_close(out[:4], stream)

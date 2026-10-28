"""Steering rows of a registered table, add-only CUDA-graph steering, and n > 1 steering."""

import pytest
import torch
from vllm import LLM, SamplingParams

from vllm_lens import SteeringRow, SteeringVector

from .conftest import LONG_PROMPT, lens_engine, relative_errors

STEER_LAYER = 2
LENS = {"steer_layers": [STEER_LAYER], "capture_layers": [STEER_LAYER]}
TABLE = "rows-test"


def prompt_rows(output) -> torch.Tensor:
    """The captured prompt rows of the steered layer."""
    acts = output.activations["residual_stream"]  # type: ignore[reportAttributeAccessIssue]
    return acts[0, : len(output.prompt_token_ids)].float()


@pytest.fixture(scope="module")
def hidden(eager_llm) -> int:
    """The hidden size of the test model."""
    return eager_llm.llm_engine.vllm_config.model_config.get_hidden_size()


@pytest.fixture(scope="module")
def table(hidden) -> torch.Tensor:
    """Three reproducible rows."""
    return torch.randn(3, hidden, generator=torch.Generator().manual_seed(0))


@pytest.fixture(scope="module")
def graph_llm():
    """CUDA graphs on, steering and capture on one layer, with the norm buffer."""
    yield from lens_engine(LENS, max_num_batched_tokens=64, enable_chunked_prefill=True)


@pytest.fixture(scope="module")
def add_only_llm():
    """CUDA graphs on, steering without the norm buffer."""
    yield from lens_engine(
        {**LENS, "steer_norm_match": False},
        max_num_batched_tokens=64,
        enable_chunked_prefill=True,
    )


def _run(engine: LLM, steering: dict, max_tokens: int = 4):
    """Generate the long prompt with capture and the given steering extra_args."""
    params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        extra_args={"output_residual_stream": [STEER_LAYER], **steering},
    )
    return engine.generate([LONG_PROMPT], params)[0]


@pytest.mark.parametrize("engine", ["eager_llm", "graph_llm", "add_only_llm"])
@pytest.mark.parametrize(
    "rows",
    [[(1, 6.0)], [(0, 4.0), (2, -3.0)]],
    ids=["one-row", "two-rows"],
)
def test_rows_match_vectors(request, engine, table, rows):
    """A row reference steers as the same vector sent in the request."""
    llm: LLM = request.getfixturevalue(engine)
    llm.set_steering_table(TABLE, table)  # type: ignore[attr-defined]
    by_row = _run(
        llm,
        {
            "apply_steering_rows": [
                SteeringRow(table=TABLE, row=r, layer_indices=[STEER_LAYER], scale=s)
                for r, s in rows
            ]
        },
    )
    by_vector = _run(
        llm,
        {
            "apply_steering_vectors": [
                SteeringVector(
                    activations=table[r : r + 1], layer_indices=[STEER_LAYER], scale=s
                )
                for r, s in rows
            ]
        },
    )
    unsteered = _run(llm, {})
    error = relative_errors(prompt_rows(by_row), prompt_rows(by_vector)).max().item()
    assert error < 1e-3, f"largest row error {error:.4f}"
    assert by_row.outputs[0].token_ids == by_vector.outputs[0].token_ids
    assert relative_errors(prompt_rows(by_row), prompt_rows(unsteered)).max() > 0.1


def test_add_only_graph_matches_the_norm_graph(graph_llm, add_only_llm, table):
    """Without the norm buffer, an addition gives the rows of the full steer op."""
    vector = SteeringVector(
        activations=table[:1], layer_indices=[STEER_LAYER], scale=8.0
    )
    full = _run(graph_llm, {"apply_steering_vectors": [vector]})
    add_only = _run(add_only_llm, {"apply_steering_vectors": [vector]})
    error = relative_errors(prompt_rows(add_only), prompt_rows(full)).max().item()
    assert error < 5e-2, f"largest row error {error:.4f}"


def test_add_only_graph_rejects_norm_match(add_only_llm, table):
    """A norm_match vector fails before generation on an add-only server."""
    vector = SteeringVector(
        activations=table[:1], layer_indices=[STEER_LAYER], norm_match=True
    )
    with pytest.raises(ValueError, match="steer_norm_match"):
        _run(add_only_llm, {"apply_steering_vectors": [vector]})


def test_unknown_table_and_cleared_table_are_rejected(graph_llm, table):
    """A row of an unregistered or cleared table fails before generation."""
    row = SteeringRow(table="missing", row=0, layer_indices=[STEER_LAYER])
    with pytest.raises(ValueError, match="not registered"):
        _run(graph_llm, {"apply_steering_rows": [row]})
    graph_llm.set_steering_table("cleared", table)  # type: ignore[attr-defined]
    graph_llm.clear_steering_table("cleared")  # type: ignore[attr-defined]
    row = SteeringRow(table="cleared", row=0, layer_indices=[STEER_LAYER])
    with pytest.raises(ValueError, match="not registered"):
        _run(graph_llm, {"apply_steering_rows": [row]})


@pytest.mark.parametrize("kind", ["rows", "vectors"])
def test_every_parallel_sample_is_steered(graph_llm, table, kind):
    """In one greedy batch, each of n samples equals a steered single request and differs from
    an unsteered one."""
    graph_llm.set_steering_table(TABLE, table)  # type: ignore[attr-defined]
    steering = (
        {
            "apply_steering_rows": [
                SteeringRow(table=TABLE, row=1, layer_indices=[STEER_LAYER], scale=8.0)
            ]
        }
        if kind == "rows"
        else {
            "apply_steering_vectors": [
                SteeringVector(
                    activations=table[1:2], layer_indices=[STEER_LAYER], scale=8.0
                )
            ]
        }
    )
    params = [
        SamplingParams(n=n, temperature=0.0, max_tokens=12, extra_args=extra)
        for n, extra in ((3, steering), (1, steering), (1, {}))
    ]
    parallel, single, unsteered = graph_llm.generate([LONG_PROMPT] * 3, params)
    samples = [list(choice.token_ids) for choice in parallel.outputs]
    assert len(samples) == 3
    assert samples == [list(single.outputs[0].token_ids)] * 3
    assert samples[0] != list(unsteered.outputs[0].token_ids)

"""Tests for projected capture via ``output_residual_stream_project``."""

import json
from types import SimpleNamespace

import pytest
import torch
from vllm import SamplingParams

from vllm_lens._activations_plugin import _check_directions, _patched_process_inputs
from vllm_lens._helpers._serialize import serialize_tensor
from vllm_lens._worker_ext import HiddenStatesExtension, _capture_rows

from .conftest import CAPTURE_LAYERS, LAYER_IDX, LONG_PROMPT, PROMPT, fake_extension


@pytest.mark.parametrize("max_tokens", [1, 4])
def test_projected_capture_matches_the_full_capture(capture_llm, max_tokens):
    """The projections and norms equal those of the full capture, at every position."""
    hidden = capture_llm.llm_engine.vllm_config.model_config.get_hidden_size()
    directions = torch.randn(3, hidden, generator=torch.Generator().manual_seed(0))
    capture_llm.set_projection("match", directions)
    extra_args = {"output_residual_stream": CAPTURE_LAYERS}
    prompts = [PROMPT, LONG_PROMPT]
    full_outputs = capture_llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=max_tokens, extra_args=extra_args),
    )
    projected_outputs = capture_llm.generate(
        prompts,
        SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            extra_args={
                **extra_args,
                "output_residual_stream_project": "match",
            },
        ),
    )
    capture_llm.clear_projection("match")

    for full_out, projected_out in zip(full_outputs, projected_outputs):
        full = full_out.activations["residual_stream"].float()  # type: ignore[reportAttributeAccessIssue]
        activations = projected_out.activations  # type: ignore[reportAttributeAccessIssue]
        assert set(activations) == {
            "residual_stream_projection",
            "residual_stream_norm",
        }
        scores = activations["residual_stream_projection"]
        norms = activations["residual_stream_norm"]
        assert scores.shape == (len(CAPTURE_LAYERS), full.shape[1], 3)
        assert norms.shape == full.shape[:2]
        assert scores.dtype == norms.dtype == torch.float32
        expected = full @ directions.T
        assert ((scores - expected).norm() / expected.norm()).item() < 1e-4
        assert ((norms - full.norm(dim=-1)).norm() / full.norm()).item() < 1e-4


@pytest.mark.parametrize(
    "schedule",
    [[(0, 10), (10, 1)], [(0, 4), (4, 4), (8, 2), (10, 1)], [(0, 6), (6, 5)]],
    ids=["single-pass", "chunked", "chunk-past-the-prompt"],
)
def test_each_step_is_projected_in_order(schedule):
    """The payload holds the projection and norm of every position once. No GPU."""
    generator = torch.Generator().manual_seed(0)
    stream = torch.randn(11, 8, generator=generator).to(torch.bfloat16)
    directions = torch.randn(2, 8, generator=generator)
    extension = fake_extension(
        {
            "r-0": {
                "output_residual_stream": [LAYER_IDX],
                "output_residual_stream_project": "dirs",
            }
        },
        directions={"dirs": directions},
    )
    for first, n_rows in schedule:
        extension.model_runner.requests["r-0"].num_computed_tokens = first
        _capture_rows(
            extension,  # type: ignore[reportArgumentType]
            LAYER_IDX,
            stream[first : first + n_rows],
            torch.tensor([0, n_rows]),
        )

    payload = HiddenStatesExtension._build_payload(extension, "r-0")["activations"]  # type: ignore[arg-type]
    torch.testing.assert_close(
        payload["residual_stream_projection"][0], stream.float() @ directions.T
    )
    torch.testing.assert_close(
        payload["residual_stream_norm"][0], stream.float().norm(dim=-1)
    )
    assert extension._projections == set()


def test_a_cleared_direction_set_skips_the_capture():
    """A request whose directions are gone is skipped and the step goes on. No GPU."""
    extension = fake_extension(
        {
            "r-0": {
                "output_residual_stream": [LAYER_IDX],
                "output_residual_stream_project": "gone",
            }
        }
    )
    _capture_rows(
        extension,  # type: ignore[reportArgumentType]
        LAYER_IDX,
        torch.zeros(4, 8, dtype=torch.bfloat16),
        torch.tensor([0, 4]),
    )
    assert extension._captured_states == {} and extension._projections == set()


def test_direction_sets_are_set_and_cleared():
    """Sets are held by name on the model device; None clears every set. No GPU."""
    worker = HiddenStatesExtension.__new__(HiddenStatesExtension)
    worker.model_runner = SimpleNamespace(model=torch.nn.Linear(2, 2))  # type: ignore[reportAttributeAccessIssue]
    worker._direction_tables = {}
    worker._hooks_installed = True
    worker.set_projection("a", json.dumps(serialize_tensor(torch.randn(2, 8).half())))
    worker.set_projection("b", json.dumps(serialize_tensor(torch.randn(3, 8))))
    assert worker._direction_tables["a"].dtype == torch.float32
    worker.clear_projection("a")
    assert set(worker._direction_tables) == {"b"}
    worker.clear_projection(None)
    assert worker._direction_tables == {}


def test_two_direction_sets_share_one_batch(capture_llm):
    """Many requests with two direction sets project with their own set."""
    hidden = capture_llm.llm_engine.vllm_config.model_config.get_hidden_size()
    generator = torch.Generator().manual_seed(1)
    sets = [torch.randn(2, hidden, generator=generator) for _ in range(2)]
    for i, directions in enumerate(sets):
        capture_llm.set_projection(f"set-{i}", directions)
    prompts = [PROMPT, LONG_PROMPT] * 12
    params = [
        SamplingParams(
            temperature=0.0,
            max_tokens=1,
            extra_args={
                "output_residual_stream": [LAYER_IDX],
                "output_residual_stream_project": f"set-{i % 2}",
            },
        )
        for i in range(len(prompts))
    ]
    full = capture_llm.generate(
        [PROMPT, LONG_PROMPT],
        SamplingParams(
            temperature=0.0,
            max_tokens=1,
            extra_args={"output_residual_stream": [LAYER_IDX]},
        ),
    )
    outputs = capture_llm.generate(prompts, params)
    for i, out in enumerate(outputs):
        expected = full[i % 2].activations["residual_stream"].float() @ sets[i % 2].T  # type: ignore[reportAttributeAccessIssue]
        scores = out.activations["residual_stream_projection"]  # type: ignore[reportAttributeAccessIssue]
        # The batch of 24 prefills in other chunks than the pair, so bf16 rows differ a
        # little; the wrong direction set would differ by order 1.
        assert ((scores - expected).norm() / expected.norm()).item() < 1e-2, (
            f"request {i}"
        )
    capture_llm.clear_projection()


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        (
            {
                "output_residual_stream_project": "known",
                "output_residual_stream_pool": "mean",
            },
            "cannot be combined",
        ),
        ({"output_residual_stream_pool": "max"}, "must be 'mean'"),
        ({"output_residual_stream_project": "unknown"}, "not registered"),
    ],
    ids=["with-pool", "unknown-pool", "unknown-name"],
)
def test_invalid_capture_is_rejected(extra, match):
    """A capture that cannot be served raises before generation. No GPU."""
    processor = SimpleNamespace(
        _lens_projections={"known"},
        model_config=SimpleNamespace(get_total_num_hidden_layers=lambda: 4),
    )
    params = SimpleNamespace(extra_args=extra, skip_reading_prefix_cache=False)
    with pytest.raises(ValueError, match=match):
        _patched_process_inputs(processor, "r", "p", params)


@pytest.mark.parametrize(
    "directions",
    [torch.randn(2, 4), torch.randn(8), torch.full((1, 8), float("nan"))],
    ids=["wrong-hidden", "one-dimensional", "non-finite"],
)
def test_invalid_directions_are_rejected(directions):
    """Directions that are not finite (N, hidden) cannot be registered. No GPU."""
    with pytest.raises(ValueError, match=r"finite \(N, 8\)"):
        _check_directions(serialize_tensor(directions), 8)

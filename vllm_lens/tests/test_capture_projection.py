"""Tests for projected capture via ``output_residual_stream_project``."""

import gc
import json
from types import SimpleNamespace

import pytest
import torch
from vllm import SamplingParams

from vllm_lens._activations_plugin import _validate_capture
from vllm_lens._helpers._serialize import serialize_tensor
from vllm_lens._worker_ext import HiddenStatesExtension, _capture_rows

from .conftest import LAYER_IDX, PROMPT, make_llm

LONG_PROMPT = "The quick brown fox jumps over the lazy dog. " * 20
LAYERS = [LAYER_IDX, LAYER_IDX + 3]


def _encode(directions: torch.Tensor) -> str:
    """The ``output_residual_stream_project`` wire form of ``directions``."""
    return json.dumps(serialize_tensor(directions))


@pytest.fixture(scope="module", params=[None, 64], ids=["single-pass", "chunked-64"])
def llm(request):
    """Offline engine, with and without a 64-token prefill chunk."""
    kwargs = {}
    if request.param is not None:
        kwargs = {
            "max_num_batched_tokens": request.param,
            "enable_chunked_prefill": True,
        }
    engine = make_llm({}, **kwargs)
    yield engine
    del engine
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("max_tokens", [1, 4])
def test_projected_capture_matches_the_full_capture(llm, max_tokens):
    """The projections and norms equal those of the full capture, at every position."""
    hidden = llm.llm_engine.vllm_config.model_config.get_hidden_size()
    directions = torch.randn(3, hidden, generator=torch.Generator().manual_seed(0))
    extra_args = {"output_residual_stream": LAYERS}
    prompts = [PROMPT, LONG_PROMPT]
    full_outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=max_tokens, extra_args=extra_args),
    )
    projected_outputs = llm.generate(
        prompts,
        SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            extra_args={
                **extra_args,
                "output_residual_stream_project": _encode(directions),
            },
        ),
    )

    for full_out, projected_out in zip(full_outputs, projected_outputs):
        full = full_out.activations["residual_stream"].float()  # type: ignore[reportAttributeAccessIssue]
        activations = projected_out.activations  # type: ignore[reportAttributeAccessIssue]
        assert set(activations) == {
            "residual_stream_projection",
            "residual_stream_norm",
        }
        scores = activations["residual_stream_projection"]
        norms = activations["residual_stream_norm"]
        assert scores.shape == (len(LAYERS), full.shape[1], 3)
        assert norms.shape == full.shape[:2]
        assert scores.dtype == norms.dtype == torch.float32
        expected = full @ directions.T
        assert ((scores - expected).norm() / expected.norm()).item() < 1e-4
        assert ((norms - full.norm(dim=-1)).norm() / full.norm()).item() < 1e-4


def _extension(extra: dict, first: int) -> SimpleNamespace:
    """A worker with one request whose next forward step starts at ``first``."""
    request = SimpleNamespace(
        sampling_params=SimpleNamespace(extra_args=extra),
        num_computed_tokens=first,
        num_prompt_tokens=10,
    )
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(num_reqs=1, req_ids=["req"]),
        requests={"req": request},
    )
    return SimpleNamespace(
        model_runner=runner, _captured_states={}, _pooled_counts={}, _projections={}
    )


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
    extension = _extension(
        {
            "output_residual_stream": [LAYER_IDX],
            "output_residual_stream_project": _encode(directions),
        },
        0,
    )
    for first, n_rows in schedule:
        extension.model_runner.requests["req"].num_computed_tokens = first
        _capture_rows(
            extension,  # type: ignore[reportArgumentType]
            LAYER_IDX,
            stream[first : first + n_rows],
            torch.tensor([0, n_rows]),
        )

    payload = HiddenStatesExtension._build_payload(extension, "req")["activations"]  # type: ignore[reportArgumentType]
    torch.testing.assert_close(
        payload["residual_stream_projection"][0], stream.float() @ directions.T
    )
    torch.testing.assert_close(
        payload["residual_stream_norm"][0], stream.float().norm(dim=-1)
    )
    assert extension._projections == {}


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ({"output_residual_stream_pool": "last"}, "cannot be combined"),
        ({"directions": torch.randn(2, 4)}, r"finite \(N, 8\)"),
        ({"directions": torch.randn(8)}, r"finite \(N, 8\)"),
        ({"directions": torch.full((1, 8), float("nan"))}, r"finite \(N, 8\)"),
    ],
    ids=["with-pool", "wrong-hidden", "one-dimensional", "non-finite"],
)
def test_invalid_projection_is_rejected(extra, match):
    """A projection that cannot be served raises before generation. No GPU."""
    directions = extra.pop("directions", torch.randn(2, 8))
    extra["output_residual_stream_project"] = _encode(directions)
    with pytest.raises(ValueError, match=match):
        _validate_capture(extra, 8)

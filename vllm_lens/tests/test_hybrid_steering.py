"""Steering and capture under CUDA graphs on a hybrid model: Qwen3.5-4B, three linear-attention
layers to one full-attention layer, behind a vision-language wrapper."""

import gc

import pytest
import torch
from vllm import LLM, SamplingParams

from vllm_lens import SteeringVector

from .conftest import LONG_PROMPT, relative_errors

MODEL = "Qwen/Qwen3.5-4B"
# A linear-attention layer and the full-attention layer after it.
STEER_LAYERS = [5, 7]
CAPTURE_LAYER = 7


def _engine(lens: dict | None):
    """A Qwen3.5-4B engine, with the plugin's CUDA-graph mode when ``lens`` is set."""
    kwargs = (
        {"additional_config": {"vllm_lens": lens}} if lens else {"enforce_eager": True}
    )
    engine = LLM(
        model=MODEL,
        dtype="auto",
        max_model_len=4096,
        gpu_memory_utilization=0.35,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
        **kwargs,
    )
    yield engine
    del engine
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def eager():
    """The forward hooks in eager mode."""
    yield from _engine(None)


@pytest.fixture(scope="module")
def graph():
    """CUDA graphs on, two steered layers and one captured layer, add-only."""
    yield from _engine(
        {
            "steer_layers": STEER_LAYERS,
            "capture_layers": [CAPTURE_LAYER],
            "steer_norm_match": False,
        }
    )


def _rows(engine: LLM, layer: int | None, hidden: int) -> torch.Tensor:
    """The captured prompt rows, steered at ``layer`` with a fixed vector, or unsteered."""
    extra: dict = {"output_residual_stream": [CAPTURE_LAYER]}
    if layer is not None:
        values = torch.randn(1, hidden, generator=torch.Generator().manual_seed(layer))
        extra["apply_steering_vectors"] = [
            SteeringVector(
                activations=values / values.norm(), layer_indices=[layer], scale=40.0
            )
        ]
    params = SamplingParams(temperature=0.0, max_tokens=2, extra_args=extra)
    output = engine.generate([LONG_PROMPT], params)[0]
    acts = output.activations["residual_stream"]  # type: ignore[reportAttributeAccessIssue]
    return acts[0, : len(output.prompt_token_ids)].float()


@pytest.mark.parametrize(
    "layer", [None, *STEER_LAYERS], ids=["unsteered", "linear", "full"]
)
def test_graph_matches_eager_on_a_hybrid_model(eager, graph, layer):
    """Graph steering and capture equal the eager hooks within 3% per row at the median, and
    steering changes the captured rows. Single rows differ more: unsteered, eager and graph
    each differ from transformers by about 1.2% at the median and up to 21% on one row."""
    hidden = eager.llm_engine.vllm_config.model_config.get_hidden_size()
    graph_rows = _rows(graph, layer, hidden)
    error = relative_errors(graph_rows, _rows(eager, layer, hidden)).median().item()
    assert error < 3e-2, f"median row error {error:.4f}"
    if layer is not None:
        assert relative_errors(graph_rows, _rows(graph, None, hidden)).median() > 0.1

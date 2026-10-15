"""Engines with different steer layers do not share a compiled graph: the settings are in
``additional_config``, part of vLLM's compile cache key. Engines boot under one cache root."""

import gc
from pathlib import Path

import pytest
import torch
from vllm import SamplingParams

from vllm_lens import SteeringVector

from .conftest import make_llm, row_error

LONG_PROMPT = "The quick brown fox jumps over the lazy dog. " * 20


def _boot_and_capture(
    cache_root: Path, env: dict[str, str], layers: list[int], extra_args: dict
) -> torch.Tensor:
    """Boot an engine under ``cache_root``, capture the prompt rows, shut it down."""
    engine = make_llm(
        {"VLLM_CACHE_ROOT": str(cache_root), **env},
        gpu_memory_utilization=0.3,
        enable_prefix_caching=False,
    )
    params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        extra_args={"output_residual_stream": layers, **extra_args},
    )
    output = engine.generate([LONG_PROMPT], params)[0]
    rows = output.activations["residual_stream"][:, : len(output.prompt_token_ids)]  # type: ignore[reportAttributeAccessIssue]
    # The next boot needs the GPU memory, and `del` does not stop the engine core.
    engine.llm_engine.engine_core.shutdown()
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return rows.float()


@pytest.fixture(scope="module")
def eager_rows(tmp_path_factory) -> dict[int, torch.Tensor]:
    """The eager forward-hook capture of layers 2, 5 and 7."""
    rows = _boot_and_capture(tmp_path_factory.mktemp("eager"), {}, [2, 5, 7], {})
    return {2: rows[0], 5: rows[1], 7: rows[2]}


def test_each_steer_layer_set_gets_its_own_graph(tmp_path, eager_rows):
    """A graph with the steer op on layer 2 is not loaded by an engine that steers layer 5."""
    hidden = eager_rows[7].shape[-1]
    vector = SteeringVector(
        activations=torch.randn(1, hidden, generator=torch.Generator().manual_seed(0)),
        layer_indices=[5],
        scale=4.0,
        norm_match=True,
    )
    steering = {"apply_steering_vectors": [vector]}
    eager = _boot_and_capture(tmp_path / "eager", {}, [7], steering)
    first_env = {"VLLM_LENS_STEER_LAYERS": "2", "VLLM_LENS_CAPTURE_LAYERS": "7"}
    _boot_and_capture(tmp_path, first_env, [7], {})
    second_env = {"VLLM_LENS_STEER_LAYERS": "5", "VLLM_LENS_CAPTURE_LAYERS": "7"}
    steered = _boot_and_capture(tmp_path, second_env, [7], steering)

    # Layer 7 shows the steering of layer 5 only if the graph has the op on layer 5.
    assert row_error(steered[0], eager[0]) < 5e-2
    assert row_error(steered[0], eager_rows[7]) > 0.1

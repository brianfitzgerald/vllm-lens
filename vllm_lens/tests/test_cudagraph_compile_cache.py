"""Tests that engines with different layer sets do not share a compiled graph.
Each test boots engines one after the other under one ``VLLM_CACHE_ROOT``."""

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


def _graph_dirs(cache_root: Path) -> set[str]:
    """The compiled-graph directories under ``cache_root``, JIT and AOT."""
    base = cache_root / "torch_compile_cache"
    jit = {p.name for p in base.glob("*") if p.name != "torch_aot_compile"}
    aot = {f"aot/{p.name}" for p in (base / "torch_aot_compile").glob("*")}
    return jit | aot


@pytest.fixture(scope="module")
def eager_rows(tmp_path_factory) -> dict[int, torch.Tensor]:
    """The eager forward-hook capture of layers 2, 5 and 7."""
    rows = _boot_and_capture(tmp_path_factory.mktemp("eager"), {}, [2, 5, 7], {})
    return {2: rows[0], 5: rows[1], 7: rows[2]}


def test_each_capture_layer_set_gets_its_own_graph(tmp_path, eager_rows):
    """A graph compiled for layer 2 is not loaded by an engine that captures layer 5."""
    _boot_and_capture(tmp_path, {"VLLM_LENS_CAPTURE_LAYERS": "2"}, [2], {})
    after_first = _graph_dirs(tmp_path)
    second = _boot_and_capture(tmp_path, {"VLLM_LENS_CAPTURE_LAYERS": "5"}, [5], {})

    assert row_error(second[0], eager_rows[5]) < 5e-2
    assert row_error(second[0], eager_rows[2]) > 0.1
    assert len(_graph_dirs(tmp_path)) > len(after_first) > 0


def test_the_same_layer_set_reuses_its_graph(tmp_path, eager_rows):
    """Two boots of one layer set share the compiled graph and agree."""
    env = {"VLLM_LENS_CAPTURE_LAYERS": "2"}
    first = _boot_and_capture(tmp_path, env, [2], {})
    after_first = _graph_dirs(tmp_path)
    second = _boot_and_capture(tmp_path, env, [2], {})

    assert _graph_dirs(tmp_path) == after_first
    # The second boot loads the graph of the first, so the rows are the same.
    assert row_error(second[0], first[0]) < 1e-4
    assert row_error(second[0], eager_rows[2]) < 5e-2


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

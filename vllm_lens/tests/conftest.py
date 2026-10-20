import gc
import multiprocessing
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

# A forked engine core can deadlock after torch CPU ops in this process.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import pytest
import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
# Compare implementations in FP32; functional/intervention tests retain BF16.
REFERENCE_DTYPE = "float32"
LAYER_IDX = 2
NUM_LAYERS = 24
PROMPT = "The future of AI is"
LONG_PROMPT = "The quick brown fox jumps over the lazy dog. " * 20
CAPTURE_LAYERS = [LAYER_IDX, LAYER_IDX + 3]
LONG_PROMPT = "The quick brown fox jumps over the lazy dog. " * 20
PROMPTS = [
    "The future of AI is",
    "Once upon a time in a land far away",
    "The quick brown fox jumps over",
    "In the beginning there was nothing but",
    "Scientists recently discovered that the universe",
    "The best way to learn programming is",
    "Deep learning models have revolutionized",
    "A long time ago in a galaxy",
    "The weather forecast for tomorrow shows",
    "Robots and humans will eventually",
]


@pytest.fixture(scope="module")
def hf_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(  # type: ignore[reportCallIssue]
        MODEL_NAME, dtype=REFERENCE_DTYPE, device_map="cuda"
    ).eval()
    yield model, tokenizer
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module", params=[None, 64], ids=["single-pass", "chunked-64"])
def capture_llm(request):
    """Offline engine, with and without a 64-token prefill chunk."""
    from vllm import LLM

    kwargs = {}
    if request.param is not None:
        kwargs = {
            "max_num_batched_tokens": request.param,
            "enable_chunked_prefill": True,
        }
    engine = LLM(model=MODEL_NAME, dtype="auto", gpu_memory_utilization=0.3, **kwargs)
    yield engine
    del engine
    gc.collect()
    torch.cuda.empty_cache()


def fake_extension(
    extras: dict[str, dict[str, Any] | None],
    directions: dict[str, torch.Tensor] | None = None,
) -> SimpleNamespace:
    """A worker stand-in whose runner holds one request per entry of ``extras``, each with
    10 prompt tokens and none computed."""
    requests = {
        req_id: SimpleNamespace(
            sampling_params=SimpleNamespace(extra_args=extra),
            num_computed_tokens=0,
            num_prompt_tokens=10,
        )
        for req_id, extra in extras.items()
    }
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(num_reqs=len(extras), req_ids=list(extras)),
        requests=requests,
    )
    return SimpleNamespace(
        model_runner=runner,
        _captured_states={},
        _pooled={},
        _captured_positions={},
        _projections={},
        _direction_tables=directions or {},
    )


def lens_engine(lens: dict[str, list[int]] | None = None, **kwargs):
    """A module fixture's offline engine, freed after; ``lens`` is its
    ``additional_config["vllm_lens"]``."""
    from vllm import LLM

    kwargs.setdefault("gpu_memory_utilization", 0.2)
    # Few graph sizes, so each engine start captures fewer graphs.
    kwargs.setdefault(
        "compilation_config", {"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64]}
    )
    if lens:
        kwargs["additional_config"] = {"vllm_lens": lens}
    engine = LLM(model=MODEL_NAME, dtype="auto", **kwargs)
    yield engine
    del engine
    gc.collect()
    torch.cuda.empty_cache()


def relative_errors(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    """``||actual - expected|| / ||expected||`` per leading index, in float32."""
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    diff = (actual.float() - expected.float()).flatten(1).norm(dim=1)
    return diff / expected.float().flatten(1).norm(dim=1)


@pytest.fixture(scope="module")
def eager_llm():
    """The reference of the CUDA-graph tests: forward hooks in eager mode."""
    yield from lens_engine(enable_prefix_caching=False)


@pytest.fixture(autouse=True, scope="module")
def _cleanup_vllm():
    yield
    # Terminate leftover vLLM EngineCore subprocesses that outlive engine.shutdown().
    for child in multiprocessing.active_children():
        child.terminate()
        child.join(timeout=5)
        if child.is_alive():
            child.kill()
    gc.collect()
    torch.cuda.empty_cache()
    if ray.is_initialized():
        ray.shutdown()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Run cross-API tests last.

    The cross-API module creates and destroys multiple vLLM engines whose
    EngineCore subprocesses may not fully terminate, holding CUDA resources.
    Running it last avoids blocking subsequent test modules; ``os._exit()``
    in ``pytest_unconfigure`` handles the final cleanup.
    """
    last: list[pytest.Item] = []
    rest: list[pytest.Item] = []
    for item in items:
        if "test_activations_cross_api" in str(item.fspath):
            last.append(item)
        else:
            rest.append(item)
    items[:] = rest + last


_exit_status = 0


def pytest_sessionfinish(session, exitstatus):
    global _exit_status
    _exit_status = exitstatus


def pytest_unconfigure(config):
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_exit_status)

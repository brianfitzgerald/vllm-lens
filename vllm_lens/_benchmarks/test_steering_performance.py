"""Opt-in steering benchmark for same-container Modal comparisons."""

import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import sys
import time
from urllib.parse import unquote, urlparse

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not os.environ.get("LENS_STEERING_BENCH"), reason="opt-in GPU benchmark"
)


@pytest.fixture(scope="module")
def installed_package():
    """Select installed code instead of pytest's source checkout."""
    distribution = importlib.metadata.distribution("vllm-lens")
    source = json.loads(distribution.read_text("direct_url.json") or "{}")
    root = (
        Path(unquote(urlparse(source["url"]).path))
        if source.get("dir_info", {}).get("editable")
        else Path(str(distribution.locate_file("")))
    )
    sys.path.insert(0, str(root))
    for name in list(sys.modules):
        if name == "vllm_lens" or name.startswith("vllm_lens.") and name != __name__:
            del sys.modules[name]
    import vllm_lens._worker_ext as worker

    assert Path(worker.__file__).is_relative_to(root)
    print(f"loaded {worker.__file__}")
    return root


@pytest.fixture(scope="module")
def engine(installed_package):
    """Create one engine with identical scheduling limits for each revision."""
    from vllm import LLM

    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    mode = os.environ.get("LENS_SMOKE_MODE", "eager")
    llm = LLM(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        dtype="bfloat16",
        seed=0,
        max_model_len=2048,
        max_num_seqs=32,
        max_num_batched_tokens=2048,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.6,
        additional_config={"vllm_lens": {"steer_layers": [2, 5, 8, 11]}}
        if mode == "graph"
        else {},
        compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64]},
    )
    yield llm
    del llm
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def results():
    """Write raw benchmark samples and the installed package identity."""
    records = []
    yield records
    distribution = importlib.metadata.distribution("vllm-lens")
    source = json.loads(distribution.read_text("direct_url.json") or "{}")
    sha = source.get("vcs_info", {}).get("commit_id", "")
    if source.get("dir_info", {}).get("editable"):
        sha = os.environ.get("LENS_SHA", "")
    import vllm_lens._worker_ext as worker

    output = {
        "worker_path": worker.__file__,
        "worker_sha256": hashlib.sha256(Path(worker.__file__).read_bytes()).hexdigest(),
        "sha": sha,
        "source": source,
        "mode": os.environ.get("LENS_SMOKE_MODE", "eager"),
        "torch_num_threads": torch.get_num_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "torch": torch.__version__,
        "vllm": importlib.metadata.version("vllm"),
        "gpu": torch.cuda.get_device_name(),
        "workloads": records,
    }
    path = Path(os.environ["LENS_SMOKE_OUT"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2))
    print(f"wrote {path}; sha={sha}; worker={worker.__file__}")


@pytest.mark.parametrize("batch", [1, 32])
@pytest.mark.parametrize(
    "length,generated",
    [(128, 1), (1024, 1), (128, 64)],
    ids=["short-prefill", "long-prefill", "decode"],
)
@pytest.mark.parametrize(
    "kind", ["none", "add", "norm", "mixed", "positional", "layers"]
)
def test_steering_workload(engine, results, batch, length, generated, kind):
    """Measure fixed token workloads after two warm-up passes."""
    from vllm import SamplingParams
    from vllm_lens import SteeringVector
    import vllm_lens._worker_ext as worker_ext

    hidden = engine.llm_engine.vllm_config.model_config.get_hidden_size()
    layers = [2, 5, 8, 11] if kind == "layers" else [2]
    values = torch.randn(
        len(layers), hidden, generator=torch.Generator().manual_seed(0)
    )
    positional = kind == "positional"
    positions = list(range(0, min(length, 256), 16)) + [length + 2]
    if positional:
        values = values[:, None].expand(-1, len(positions), -1).clone()
    vector = SteeringVector(
        activations=values,
        layer_indices=layers,
        scale=0.5,
        norm_match=kind in ("norm", "mixed"),
        position_indices=positions if positional else None,
    )
    params = [
        SamplingParams(
            temperature=0,
            max_tokens=generated,
            ignore_eos=True,
            extra_args={"apply_steering_vectors": [vector]}
            if kind != "none" and (kind != "mixed" or i % 2 == 0)
            else None,
        )
        for i in range(batch)
    ]
    prompts = [
        {"prompt_token_ids": [1000 + (i + j) % 10000 for j in range(length)]}
        for i in range(batch)
    ]
    samples = []
    for repeat in range(9):
        started = time.perf_counter()
        outputs = engine.generate(prompts, params, use_tqdm=False)
        elapsed = time.perf_counter() - started
        assert len(outputs) == batch
        assert all(len(output.outputs[0].token_ids) == generated for output in outputs)
        if repeat >= 2:
            samples.append(elapsed)
    memory = engine.collective_rpc(
        lambda worker: (
            torch.cuda.memory_allocated(),
            torch.cuda.max_memory_allocated(),
            hashlib.sha256(Path(worker_ext.__file__).read_bytes()).hexdigest(),
            worker_ext.__file__,
            torch.get_num_threads(),
        )
    )
    driver = hashlib.sha256(Path(worker_ext.__file__).read_bytes()).hexdigest()
    assert all(row[2] == driver for row in memory), memory
    median = statistics.median(samples)
    result = {
        "case": f"{kind}/{batch}/{length}/{generated}",
        "samples_s": samples,
        "median_s": median,
        "tokens_per_s": batch * (length + generated) / median,
        "gpu_memory_bytes": [row[:2] for row in memory],
        "workers": [
            {"worker_sha256": row[2], "path": row[3], "torch_num_threads": row[4]}
            for row in memory
        ],
        "kv_cache_blocks": engine.llm_engine.vllm_config.cache_config.num_gpu_blocks,
    }
    results.append(result)
    print(json.dumps(result))

# vLLM-Lens Benchmarks

Note these are designed to run on the [Isambard cluster](https://docs.isambard.ac.uk/specs/#system-specifications-isambard-ai-phase-2), so they won't work out-of-the-box unless you have access to it.


Install the public benchmark dependencies with `uv sync --group benchmarking`.
The launcher uses Hugging Face's public `datasets` and `huggingface_hub` APIs
for dataset/model downloads. No private repositories are required.

Supply your Apptainer images explicitly in a directory accessible to the compute
nodes, named `<container_name>.sif` (for example `vllm-0.18.0.sif`). The benchmark
configuration in `run_all.py` specifies which images each comparison needs;
these historical comparison versions are independent of the library's current
validated vLLM pin.

```bash
uv run python vllm_lens/_benchmarks/run_all.py \
  --container-dir /path/to/images --benchmarks pure-vllm --dry-run
```

`VLLM_BENCHMARK_CONTAINER_DIR` can also supply the image directory. Dry runs
validate image paths and print submissions without downloading models/datasets
or submitting jobs. Real runs prefetch weights/data before submitting through
Slurm; gated models still require your Hugging Face access.

## Steering regression check

The steering benchmark needs a CUDA GPU and the pinned vLLM version. It tests the installed package, with two warm-up passes and seven measured passes per workload.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  LENS_STEERING_BENCH=1 LENS_SMOKE_MODE=eager \
  LENS_SMOKE_OUT=/tmp/steering-candidate.json \
  pytest vllm_lens/_benchmarks/test_steering_performance.py -s
```

Use `LENS_SMOKE_MODE=graph` to test CUDA graphs. Run candidate, baseline, then candidate again in the same container, with the same benchmark file and settings. Each result records the package and worker code identity, raw times, throughput, memory, and KV-cache capacity.

```bash
python scripts/compare_steering_performance.py \
  /tmp/steering-candidate.json /tmp/steering-baseline.json /tmp/steering-repeat.json
```

The comparison fails if the baseline loaded the candidate code, the workload sets differ, or the repeated candidate median is more than 5% slower than the baseline. The first candidate run also warms the container. Repeat flagged comparisons twice before accepting a regression.

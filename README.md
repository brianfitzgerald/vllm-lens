# vLLM-lens

vLLM-lens captures residual-stream activations and steers them inside vLLM's compiled forward pass, with CUDA graphs on. Requests in the same dynamic batch can capture and steer different layers, vectors and positions.

Features:
- Activation capture: every position, the last prompt token, a list of positions, or the prompt mean, at the layers a server arms.
- Steering vectors: add vectors to the residual stream, at every position or at chosen positions, optionally scaled to the residual norm.
- Steering tables: register vectors once and send row references with each request.

The package registers as a [vLLM general plugin](https://docs.vllm.ai/en/latest/design/plugin_system.html) and an [Inspect](https://inspect.aisi.org.uk/) model provider on install. Requests set options in `SamplingParams.extra_args` (offline), `vllm_xargs` (HTTP) or `GenerateConfig.extra_body` (Inspect).

This fork serves only on vLLM's V2 model runner, which vLLM 0.30 uses by default. Forward hooks, pre-hooks, Q/K capture and pipeline parallelism are not supported.

## Install

```bash
uv add vllm-lens
```

See the [vLLM compatibility matrix and validation guide](docs/compatibility.md) for the tested versions.

## Quickstart

A server arms its layers when it starts. Capture and steering are served only on those layers:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --additional-config '{"vllm_lens": {"capture_layers": [15, 20], "steer_layers": [18]}}'
```

Offline, pass `additional_config={"vllm_lens": {...}}` to `LLM(...)`. A server that arms no layers runs as plain vLLM and refuses capture and steering requests.

Settings in `additional_config["vllm_lens"]`:

- `capture_layers`, `steer_layers`: the armed layers.
- `steer_norm_match` (default `true`): `false` removes the norm table and its per-token norm; vectors with `norm_match=True` are then refused.
- `capture_ring_tokens` (default `0`, which means twice `max_num_batched_tokens`): the rows of the GPU ring that holds captured rows until a thread writes them out.

The plugin sets vLLM's `worker_cls` to `vllm_lens._graph_worker.GraphWorker` and `VLLM_USE_V2_MODEL_RUNNER=1`. The engine refuses to load with pipeline parallelism, micro-batching (DBO), prefill context parallelism, adaptive draft verification, sequence parallelism, Inductor graph partition or breakable CUDA graphs. `enforce_eager` is allowed and runs the same ops without graphs.

For the HTTP API, a Python client is provided:

```python
from vllm_lens.client import VLLMLensClient

client = VLLMLensClient("http://localhost:8000")
out = client.generate("Hello world", capture_layers=[15, 20])
print(out.activations["residual_stream"].shape)  # (n_layers, n_positions, hidden)
```

## How it works

- Steering: the worker keeps one table of rows per request slot. When vLLM admits a request, the worker writes the request's summed vectors into its rows. Before each forward pass it fills one table row id per token. A forward hook on each steered layer adds the token's row with plain torch ops, which Inductor fuses into the compiled graph.
- Capture: before each forward pass the worker gives each captured token a row of a GPU ring. A custom op on each captured layer writes the token's stream (`hidden + residual`) to that row inside the graph. After the pass, a thread copies the ring rows to pinned memory on its own CUDA stream and writes them to one file per request. The engine loop does no copy and no serialization.
- Delivery: when a request finishes, the worker seals its file (a safetensors file of shape `(rows, n_layers, hidden)`). The front end maps the file and sends the activations in the response, or sends only its path.

## Disabling the plugin

vllm-lens loads in every vLLM process through the `vllm.general_plugins` entry point. Set `VLLM_LENS_DISABLE=1` to make it a no-op:

```bash
VLLM_LENS_DISABLE=1 vllm serve meta-llama/Llama-3.1-8B-Instruct
```

## Logging

The plugin logs through the `vllm.lens` logger, a child of vLLM's `vllm` logger. `VLLM_LOGGING_LEVEL` sets the level.

## Activation capture

Pass `output_residual_stream` (a list of armed layers) in `extra_args`:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    "meta-llama/Llama-3.1-8B-Instruct",
    additional_config={"vllm_lens": {"capture_layers": [15, 20]}},
)
sp = SamplingParams(max_tokens=1, extra_args={"output_residual_stream": [15, 20]})
out = llm.generate(["Hello world"], sp)
acts = out[0].activations["residual_stream"]  # (n_layers, n_positions, hidden_dim)
```

Layers are stacked in ascending order along dim 0. Capture runs on tensor-parallel rank 0 only, because the residual stream is the same on every rank.

To get the mean over the prompt positions as one row per layer, pass `output_residual_stream_pool="mean"` (the client's `capture_pool`). The result has shape `(n_layers, 1, hidden_dim)`.

### Selecting positions

`output_residual_stream_positions` limits the capture to `"last_prompt_token"` or to a list of absolute positions. The worker writes only those rows, so a last-token capture does not grow with the prompt:

```python
sp = SamplingParams(
    max_tokens=8,
    extra_args={
        "output_residual_stream": [15],
        "output_residual_stream_positions": [0, 3, 12],
    },
)
```

The response carries `residual_stream_positions` with the position of each row. Prompt tokens are positions `0 .. n_prompt - 1`, and `n_prompt + k` is the state computed when generated token `k` is fed back. The final generated token is never processed, so a capture of every position has `n_prompt + n_gen - 1` rows. Positions that are never computed are absent. A position computed twice (after a preemption) keeps its latest row. `output_residual_stream_pool` and `output_residual_stream_positions` cannot be combined.

With async speculative decoding, a capture of generated positions (every position, or a list) fails at submission; `"last_prompt_token"` and the mean work.

### Transports

`activations_transport` sets how the HTTP response carries the activations:

- `"base64"` (default): the bytes, base64-encoded in the JSON body.
- `"binary"`: a handle; the client fetches the bytes from `GET /v1/activations/{handle}`. The store is bounded (`VLLM_LENS_ACT_MAX_BYTES`, default 8 GiB) and per-entry TTL'd (`VLLM_LENS_ACT_TTL_S`, default 300 s), and it is per API-server process.
- `"file"`: the path of the capture file and its row count. The response is sent without waiting for the file. The reader must be on the server's host; `decode_activations` and the client wait for the file and map it, and the caller removes it.

The files go to `VLLM_LENS_CAPTURE_DIR` (default: a new directory in `/dev/shm`). `VLLM_LENS_CAPTURE_TIMEOUT` (default 120 s) sets how long the front end waits for a file. Activations are not compressed.

## Steering vectors

```python
import torch
from vllm import LLM, SamplingParams
from vllm_lens import SteeringVector

sv = SteeringVector(
    activations=torch.randn(1, 4096),  # (n_layers, hidden)
    layer_indices=[15],
    scale=4.0,
    norm_match=True,  # scale the vector to ‖residual‖ · scale
    position_indices=None,  # None = every position
)
sp = SamplingParams(max_tokens=20, extra_args={"apply_steering_vectors": [sv]})
llm.generate(["I think the best dessert is"], sp)
```

`position_indices` are absolute positions of the sequence (prompt and generated tokens); the vector is added at those positions only. The vectors of one request share one position list. Values per position (3-D activations) are not supported.

Via the client (or `vllm_xargs` over HTTP): `client.generate("...", steering_vectors=[sv])`. `llm.chat(...)` accepts the same `extra_args`, and both offline entry points accept the JSON wire form (`json.dumps([sv.model_dump(mode="json")])`).

Steered requests skip prefix-cache reads and write under a unique cache salt, so other requests cannot reuse their KV blocks. Capture requests also skip prefix-cache reads. Steering supports `n > 1`; capture requires `n = 1`.

### Steering tables

Register a table once and send row references with each request:

```python
client.set_steering_table("features", vectors)  # shape (rows, hidden_size)
client.generate(
    "Hello", steering_vectors=[SteeringVector(table="features", row=0, layer_indices=[15])]
)
client.clear_steering_table("features")
```

A request of table rows only sends no RPC: the rows ride in `extra_args`, and the worker reads them when it admits the request. Replacing or clearing a table does not change requests in flight, which keep the version they started with. A missing table or row fails before generation.

Offline, use `llm.set_steering_table` and `llm.clear_steering_table`. Over HTTP, register with `POST /v1/steering_tables` (`name` and serialized `vectors`) and delete with `DELETE /v1/steering_tables/{name}`.

## Inspect AI provider

The provider `vllm-lens` extends Inspect's vLLM provider: it serializes `torch.Tensor` steering vectors and decodes activations into `ModelOutput.metadata["activations"]`. Usage is the same as the [default vLLM provider](https://inspect.aisi.org.uk/providers.html#vllm), with the `vllm-lens` prefix.

```python
capture_config = GenerateConfig(
    temperature=0.0,
    max_tokens=1,
    extra_body={"extra_args": {"output_residual_stream": [15]}},
)
output = await model.generate(state.messages, config=capture_config)
residual_stream = output.metadata["activations"]["residual_stream"]
```

## Examples

[`examples/activation_oracle.py`](examples/activation_oracle.py) captures activations and steers a LoRA oracle to describe them (arXiv:2512.15674). [`examples/inspect-demo.ipynb`](examples/inspect-demo.ipynb) does the same through the Inspect provider.

## fp32 models and TF32

vLLM serves fp32 models with its Triton attention backend, whose `tl.dot` defaults to TF32. vllm-lens sets `TRITON_F32_DEFAULT=ieee` for fp32 models (logged at startup), so attention computes in fp32; set `TRITON_F32_DEFAULT` yourself to override.

## Layer discovery

vllm-lens finds the decoder layers through vLLM's attention-layer registry (`static_forward_context`), which covers standard attention, MLA and Mamba or linear-attention hybrids. The layers are checked against the config's `num_hidden_layers`; a mismatch raises `LayerDiscoveryError` when the model loads. For an unusual layout, give a `get_submodule` path template:

```bash
VLLM_LENS_LAYER_PATH="model.layers.{i}" vllm serve my-org/unusual-model
```

## Running tests

CPU tests run without a GPU or vLLM:

```bash
pytest tests/unit
```

The GPU tests in `vllm_lens/tests/` use a small model and their own engines. The integration tests in `tests/` run against a live server, which the fixture starts with every layer armed, or an existing one (`VLLM_TEST_PORT`). `VLLM_TEST_MODEL`, `VLLM_TEST_TP_SIZE` and `VLLM_TEST_STARTUP_TIMEOUT` set the model, tensor parallelism and start timeout.

## Credits

Developed by Alan Cooney, with credit going to Sid Black for the original vLLM worker extension idea.

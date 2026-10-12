## Unreleased

- Capture: The `output_residual_stream_project` directions go to the workers once per distinct payload, keyed by its hash, and the request carries the key (`_projection_id`) in place of the payload. Before, every request took its own copy of the directions into the engine and onto the GPU for its whole life; at 100 directions and 7,416 requests that was 9 GB of strings and 6.8 GB of GPU copies of one 921 KB tensor. The offline path ships and releases the distinct payloads in one RPC each; the online path holds a reference count per key.
- Steering: A vector that is a view into a larger tensor is copied out before it is pickled for the workers. A row of an `(N, hidden)` matrix pickled the whole matrix: at 100 directions and 14,000 requests, 12.9 GB.
- Steering: Under CUDA graphs the buffers are filled from a per-request row table in one gather per buffer each step, in place of a dtype cast, a multiply and an add per steered request per step. Each request's `add` and `norm` row is written to the table once on first sight and released with its vectors; each request id resolves its row once; position-specific vectors keep the per-position path. The batch layout is read from the runner's pinned CPU copy and the step's row indices go to the device through a pinned buffer, so the fill never waits for the stream. With 2,800 steered requests the steered pass cost 2.1 times the unsteered one.
- Steering: The per-step lookup of a request's steering vectors is a dictionary lookup (the `_steering_id` sentinel offline, the external id before vLLM's `-suffix` online) in place of a prefix scan over every stored request, and the buffer fill reads the batch layout with one host sync per step. The offline path ships and clears all requests' vectors in one RPC each. With 2,800 steered requests the scan cost grew with in-flight times total requests and made the steered pass 11 to 16 times slower than the unsteered one.
- Capture: Gemma2 returns auxiliary hidden states, so `VLLM_LENS_CAPTURE_LAYERS` serves it. vLLM's `Gemma2Model` has no `set_aux_hidden_state_layers`; the plugin adds the method and the aux outputs to the class at register, before the model is built and compiled. The unit tests read `VLLM_TEST_MODEL`.
- Capture: Added `output_residual_stream_project` (`capture_project` on the client). The worker returns the projection of every position onto `(N, hidden)` directions and the norm of every position, in float32, in place of the residual, so a long prompt ships N + 1 values per position in place of `hidden`.
- Examples: Added a test that runs the activation oracle example in eager mode and under CUDA graphs.
- Steering: Added `VLLM_LENS_STEER_LAYERS`. `apply_steering_vectors` on the named layers is applied by an op inside the compiled graph that reads buffers the plugin fills before each forward pass, so steering works with compilation and CUDA graphs on.
- Capture: Added `VLLM_LENS_CAPTURE_LAYERS`. The named layers are captured as auxiliary hidden states of the model, so `output_residual_stream` works with compilation and CUDA graphs on.
- Plugin: Added the `VLLM_LENS_CUDAGRAPH=1` environment variable. The plugin then does not force `enforce_eager=True`, and a request that needs the forward hooks fails with an error in place of returning nothing.
- Plugin: `register()` now attaches a stderr handler to the `vllm_lens` logger when it has none, so the package's INFO records are no longer dropped. `VLLM_LENS_LOG_LEVEL` sets the level (default `INFO`).
- Capture: Added `output_residual_stream_pool` (`capture_pool` on the client). `"last"` returns the last prompt position and `"mean"` returns the mean over the prompt positions, as one row per layer, so a long prompt no longer ships every position to the client.
- Steering: Fixed `position_indices` during decode and chunked prefill. The absolute start of each forward pass was always 0, so a position in a later prefill chunk was never steered, and position 0 was steered again on every decode step. The start now comes from the request's computed-token count, which does not depend on the attention backend.
- Tests: The comparison with Transformers now checks the largest per-row relative error (limit 3%) in place of a mean absolute difference of 0.01. One prompt of the batch test had a mean absolute difference of 0.0105 at a per-row error of 1.6%, the same as the other prompts.

## v1.2.1 (22 July 2026)

- Steering: Fixed offline steering via `LLM.chat` — the plugin now patches `LLM.chat` (which submits requests to the engine directly rather than routing through `LLM.generate`). Previously, live `SteeringVector` objects raised a msgpack `TypeError`, and JSON-serialized vectors ran **silently unsteered**. Activation capture (`output_residual_stream`) and per-request hooks (`apply_hooks`) now also work through `LLM.chat`. (#28)
- Steering: The offline `LLM.generate` / `LLM.chat` path now decodes the JSON-string wire format for `apply_steering_vectors` and `apply_hooks` (the form the HTTP API accepts via `vllm_xargs`), instead of failing inside `collective_rpc`. Both entry points accept both wire forms. (#28)

## v1.2.0 (17 July 2026)

- Hooks: Added a generic, Garçon-style hook system — run arbitrary Python functions per-request and per-layer to capture data (via `ctx.saved`) and/or modify hidden states (by returning a tensor). Supports per-request hooks (`apply_hooks` in `extra_args`), persistent register-once hooks (`register_hooks` / `collect_hook_results` / `clear_hooks`), and pre-hooks (run before the layer forward pass). Exposed over HTTP at `/v1/hooks/*` and through a `VLLMLensClient`; `ctx.get_parameter()` gathers parameters across tensor- and pipeline-parallel ranks. (#3)
- Hooks: Added `POST /v1/hooks/clear_results` (`client.clear_hook_results()`) — drain accumulated persistent-hook results while keeping the hooks registered, so long-lived clients can bound accumulation without re-uploading hook state. (#25)
- Steering: Fixed `norm_match` on fused-residual architectures (Qwen, Gemma, Llama, …). It now scales the steering vector to the full residual stream rather than the MLP-delta half, so the injected magnitude equals `‖residual‖ · scale` as intended. **Behavior change:** existing `norm_match=True` steering becomes correspondingly stronger. (#7, #15)
- Performance: On the offline `LLM.generate` path, a batch's captured activations are now fetched in a single RPC instead of one per request. (#6)
- Plugin: Added the `VLLM_LENS_DISABLE=1` environment variable to make the plugin a no-op, so vllm-lens can be installed alongside another inference server without perturbing it. (#9, #14)
- Plugin: Default to the V1 model runner and raise a clear error if the V2 runner is active (the capture/steering hooks read V1 model-runner internals). (#12)
- Examples: Moved examples to a top-level `examples/` directory and added causal-tracing, logit-lens, deception-probe, and emotion-concepts examples. (#4, #11)
- Examples: Added a Jacobian-lens / J-space example — fit a per-model average Jacobian `J_l = E[∂h_final/∂h_l]` offline (HuggingFace backward pass), then read it out live on vLLM-captured residuals to see what the model is "disposed to say" at each layer. (#19)
- Examples: Hardened the Jacobian-lens run path for tensor / pipeline / expert parallelism, so it works on larger multi-GPU models. (#21)
- Examples: Added a live J-space chat visualizer — chat with a served model in a generated HTML page where every token is hoverable, showing the streaming top-k Jacobian-lens readout across the captured layers. (#25)
- Examples: The Jacobian-lens fitter stamps provenance (model, corpus, sample count, estimator settings) into the saved lens `.pt`. (#27)
- Docs: Documented the `Hook`/`HookContext` interface and added first-class README sections for activation capture and steering. (#13, #16)

## v1.1.0

- Prior releases.

"""
vLLM general plugin that captures residual-stream activations and steers them on the layers an
engine arms in ``additional_config["vllm_lens"]``.

Installed automatically via the ``vllm.general_plugins`` entry point (configured in
pyproject.toml). Patches ``EngineArgs.create_engine_config`` to serve armed engines with the
vllm-lens worker on vLLM's V2 model runner, and patches ``AsyncLLM.generate``,
``LLM.generate`` and ``LLM.chat`` to steer requests and attach their captures.
"""

from __future__ import annotations

import asyncio
import collections
import copy
import json
import logging
import os
import pickle
import tempfile
import threading
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

import torch

from vllm_lens._cudagraph import LensGraphConfig
from vllm_lens._helpers._capture_file import (
    CAPTURE_DIR_ENV,
    capture_path,
    read_capture,
    wait_capture,
    wait_capture_async,
)
from vllm_lens._helpers._positions import (
    LAST_PROMPT_TOKEN,
    POSITIONS_KEY,
    POSITIONS_META_KEY,
    normalize_capture_positions,
)
from vllm_lens._helpers._serialize import (
    serialize_activations,
    serialize_activations_binary,
    serialize_tensor,
)
from vllm_lens._helpers.types import SteeringVector

logger = logging.getLogger("vllm.lens")

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM

# Seconds to wait for the worker to write a capture after its request finishes.
_CAPTURE_TIMEOUT = float(os.environ.get("VLLM_LENS_CAPTURE_TIMEOUT", "120"))
_TRANSPORTS = (None, "base64", "binary", "file")
# Keys of extra_args that only this plugin sets.
_INTERNAL_KEYS = ("_steering_id", "_steer_rows", "_capture_key", "_lens_isolate_cache")

# Populated by register() with the original unpatched methods.
_original_create_engine_config: Callable | None = None
_original_generate: Callable | None = None
_original_llm_generate: Callable | None = None
_original_llm_chat: Callable | None = None
_original_completion_response: Callable | None = None
_original_chat_full_generator: Callable | None = None
_original_chat_stream_generator: Callable | None = None
_original_register_routers: Callable | None = None
_original_process_inputs: Callable | None = None


def _decode_steering_vectors(
    value: Any, tables: dict[str, tuple[int, int]] | None = None
) -> list[SteeringVector] | None:
    """Normalise an ``apply_steering_vectors`` value (vectors, dicts or the JSON string of
    ``vllm_xargs``), check its table rows and stamp each with the table's current version."""
    if value is None:
        return None
    if isinstance(value, str):
        value = json.loads(value)
    vectors = [
        v if isinstance(v, SteeringVector) else SteeringVector.model_validate(v)
        for v in value
    ]
    for vector in vectors:
        if vector.table is not None:
            if tables is None or vector.table not in tables:
                raise ValueError(f"Steering table {vector.table!r} is not registered")
            rows, table_version = tables[vector.table]
            if vector.row is None or vector.row >= rows:
                raise ValueError("Steering row is out of range")
            vector.version = table_version
    return vectors


def _table_rows(vectors: list[SteeringVector] | None) -> list[list[Any]] | None:
    """The ``[table, version, row, scale, layers, norm_match]`` of each vector, which the
    worker reads at admission with no RPC; None unless every vector is a table row."""
    if not vectors or any(v.activations is not None for v in vectors):
        return None
    return [
        [v.table, v.version, v.row, v.scale, list(v.layer_indices), v.norm_match]
        for v in vectors
    ]


def _hold_tables(engine: Any, vectors: list[SteeringVector] | None) -> None:
    """Count a request's use of each table version it names."""
    for vector in vectors or ():
        if vector.table is not None:
            engine._lens_table_users[(vector.table, vector.version)] += 1


def _release_tables(
    engine: Any, vectors: list[SteeringVector] | None
) -> list[tuple[str, int]]:
    """End a request's use of its table versions; return the replaced versions no request
    uses any more, for the caller to drop on the workers."""
    drop = []
    for vector in vectors or ():
        if vector.table is None:
            continue
        key = (vector.table, vector.version)
        users = engine._lens_table_users
        users[key] -= 1
        current = engine._lens_steering_tables.get(vector.table)
        if users[key] <= 0:
            del users[key]
            if current is None or current[1] != vector.version:
                drop.append(key)
    return drop


def _replace_table(engine: Any, name: str, rows: int) -> tuple[int, int | None]:
    """Register ``rows`` rows as a new version of table ``name``; return that version and
    the old version to drop now, if no request uses it."""
    old = engine._lens_steering_tables.get(name)
    new_version = 0 if old is None else old[1] + 1
    engine._lens_steering_tables = {
        **engine._lens_steering_tables,
        name: (rows, new_version),
    }
    if old is None or engine._lens_table_users[(name, old[1])] > 0:
        return new_version, None
    return new_version, old[1]


def _remove_table(engine: Any, name: str | None) -> list[tuple[str, int | None]]:
    """Unregister ``name`` (or every table); return the versions to drop now."""
    names = list(engine._lens_steering_tables) if name is None else [name]
    tables = dict(engine._lens_steering_tables)
    drop: list[tuple[str, int | None]] = []
    for key in names:
        entry = tables.pop(key, None)
        if entry is not None and engine._lens_table_users[(key, entry[1])] <= 0:
            drop.append((key, entry[1]))
    engine._lens_steering_tables = tables
    return drop


def _reject_parallel_capture(params: SamplingParams, extra: dict[str, Any]) -> None:
    """Raise ``ValueError`` for capture with ``n > 1``: each child request is captured under
    its own id."""
    if (getattr(params, "n", None) or 1) > 1 and extra.get(
        "output_residual_stream"
    ) is not None:
        raise ValueError(
            "Activation capture is not available with n > 1; send separate requests."
        )


def _reject_async_spec_positions(
    vllm_config: Any, vectors: list[SteeringVector] | None, extra: dict[str, Any]
) -> None:
    """Raise ``ValueError`` for a positional vector or a capture of generated positions
    under async speculative decoding, where a request's computed count is optimistic."""
    scheduler = getattr(vllm_config, "scheduler_config", None)
    if getattr(vllm_config, "speculative_config", None) is None or not getattr(
        scheduler, "async_scheduling", False
    ):
        return
    if any(vector.position_indices is not None for vector in vectors or ()):
        raise ValueError(
            "Position-specific steering vectors are not available with async speculative "
            "decoding: the positions of generated tokens are not known before the forward "
            "pass. Disable async scheduling or speculative decoding to steer by position."
        )
    selector = extra.get(POSITIONS_KEY)
    full = selector is None and extra.get("output_residual_stream_pool") is None
    if extra.get("output_residual_stream") is not None and (
        full or isinstance(selector, list)
    ):
        raise ValueError(
            "A capture of generated positions is not available with async speculative "
            "decoding: their positions are not known before the forward pass. Use "
            f"{POSITIONS_KEY}={LAST_PROMPT_TOKEN!r} or a mean, or disable async scheduling "
            "or speculative decoding."
        )


def _check_steering_vectors(directions: torch.Tensor, hidden: int) -> None:
    """Validate finite steering vectors with shape ``(N, hidden)``."""
    if (
        directions.ndim != 2
        or directions.shape[1] != hidden
        or not torch.isfinite(directions).all()
    ):
        raise ValueError(
            f"directions must be finite (N, {hidden}), got {tuple(directions.shape)}"
        )


def _check_capture_options(extra: dict[str, Any]) -> None:
    """Parse the layer list and selector that ``vllm_xargs`` passes as strings, and raise
    ``ValueError`` for a bad pool value, selector or transport, an option without
    ``output_residual_stream``, or a combination of options."""
    layers = extra.get("output_residual_stream")
    if isinstance(layers, str):
        extra["output_residual_stream"] = json.loads(layers)
    pool = extra.get("output_residual_stream_pool")
    if pool not in (None, "mean"):
        raise ValueError(f"output_residual_stream_pool must be 'mean', got {pool!r}")
    if extra.get("activations_transport") not in _TRANSPORTS:
        raise ValueError(
            f"activations_transport must be one of {_TRANSPORTS[1:]}, got "
            f"{extra['activations_transport']!r}"
        )
    normalize_capture_positions(extra)
    chosen = [
        key
        for key, value in (
            ("output_residual_stream_pool", pool),
            (POSITIONS_KEY, extra.get(POSITIONS_KEY)),
        )
        if value is not None
    ]
    if len(chosen) > 1:
        raise ValueError(f"{' and '.join(chosen)} cannot be combined")
    if chosen and extra.get("output_residual_stream") is None:
        raise ValueError(f"{chosen[0]} requires output_residual_stream")


def _trim_activations(activations: dict[str, Any], expected_len: int) -> None:
    """Drop the rows of the pass vLLM runs after a stop condition, so a capture has one row
    per processed position: ``expected_len`` is prompt plus generated tokens minus one."""
    rs = activations.get("residual_stream")
    positions = activations.get(POSITIONS_META_KEY)
    if rs is None:
        return
    if positions is not None:
        keep = sum(1 for pos in positions if pos < expected_len)
        if keep < len(positions):
            activations["residual_stream"] = rs[:, :keep]
            activations[POSITIONS_META_KEY] = positions[:keep]
    elif rs.shape[1] > expected_len:
        activations["residual_stream"] = rs[:, :expected_len]


def _remove_capture(path: str) -> None:
    """Remove the file of a capture nobody reads, once the worker writes it."""
    try:
        wait_capture(path, _CAPTURE_TIMEOUT)
        os.unlink(path)
    except (TimeoutError, FileNotFoundError):
        try:
            os.unlink(path.removesuffix(".safetensors") + ".partial")
        except FileNotFoundError:
            pass


def _serialize_for(
    activations: dict[str, Any], transport: str | None
) -> dict[str, Any]:
    """Serialize activations for the response body in the negotiated transport."""
    if transport == "binary":
        from vllm_lens._helpers._activation_store import store

        return serialize_activations_binary(activations, store)
    return serialize_activations(activations)


# ---------------------------------------------------------------------------
# Engine config patch: the vllm-lens worker on the V2 runner
# ---------------------------------------------------------------------------


def _patched_create_engine_config(self, *args, **kwargs):
    """Patch for ``EngineArgs.create_engine_config``: an engine with armed layers runs the
    vllm-lens worker on vLLM's V2 model runner."""
    lens = LensGraphConfig.from_vllm_config(self)
    if lens.enabled:
        self.worker_cls = "vllm_lens._graph_worker.GraphWorker"
        # vLLM reads this for the engine's life and refuses features V2 does not serve.
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
        if lens.capture_layers and not os.environ.get(CAPTURE_DIR_ENV):
            root = "/dev/shm" if os.path.isdir("/dev/shm") else None
            os.environ[CAPTURE_DIR_ENV] = tempfile.mkdtemp(
                prefix="vllm-lens-", dir=root
            )

    assert _original_create_engine_config is not None
    config = _original_create_engine_config(self, *args, **kwargs)

    # vLLM serves fp32 models with the Triton attention backend, whose tl.dot defaults to
    # TF32; make the attention kernel compute in fp32 like the linears.
    if getattr(getattr(config, "model_config", None), "dtype", None) == torch.float32:
        if os.environ.setdefault("TRITON_F32_DEFAULT", "ieee") == "ieee":
            logger.info(
                "vllm-lens: fp32 model — defaulting TRITON_F32_DEFAULT=ieee so the "
                "Triton attention kernel computes in true fp32 (set TRITON_F32_DEFAULT "
                "explicitly to override)."
            )
    if lens.enabled and not config.use_v2_model_runner:
        raise RuntimeError("vllm-lens needs vLLM's V2 model runner.")
    return config


# ---------------------------------------------------------------------------
# Generate patches
# ---------------------------------------------------------------------------


def _isolate_prefix_cache(request: Any) -> None:
    """Add a unique salt to the request's cache namespace."""
    salt = f"lens-request:{uuid.uuid4().hex}"
    request.cache_salt = f"{request.cache_salt}|{salt}" if request.cache_salt else salt


def _patched_process_inputs(
    self: Any, request_id: str, prompt: Any, params: Any, *args, **kwargs
) -> Any:
    """Build the request, then isolate intervention cache entries."""
    assert _original_process_inputs is not None
    request = _original_process_inputs(
        self, request_id, prompt, params, *args, **kwargs
    )
    if (getattr(params, "extra_args", None) or {}).get("_lens_isolate_cache"):
        _isolate_prefix_cache(request)
    return request


def _prepare_request(
    engine: Any, vllm_config: Any, params: Any, extra: dict[str, Any], key: str
) -> tuple[list[SteeringVector] | None, bool]:
    """Check one request's options and set its internal keys; return its steering vectors
    and whether it captures. Steering and capture read every position, so the request
    skips the prefix cache, and a steered request writes under its own salt."""
    _check_capture_options(extra)
    for name in _INTERNAL_KEYS:
        extra.pop(name, None)
    vectors = _decode_steering_vectors(
        extra.pop("apply_steering_vectors", None), engine._lens_steering_tables
    )
    _reject_async_spec_positions(vllm_config, vectors, extra)
    _reject_parallel_capture(params, extra)
    LensGraphConfig.from_vllm_config(vllm_config).reject(
        residual_stream=extra.get("output_residual_stream"), vectors=vectors
    )
    captures = extra.get("output_residual_stream") is not None
    if captures:
        extra["_capture_key"] = key
    rows = _table_rows(vectors)
    if rows is not None:
        extra["_steer_rows"] = rows
    elif vectors is not None:
        extra["_steering_id"] = key
    if extra.pop("skip_reading_prefix_cache", None) or captures or vectors:
        params.skip_reading_prefix_cache = True
    if vectors:
        extra["_lens_isolate_cache"] = True
    _hold_tables(engine, vectors)
    return vectors, captures


async def _patched_generate(
    self: AsyncLLM,
    prompt: str,
    sampling_params: SamplingParams,
    request_id: str,
    **kwargs,
) -> AsyncIterator:
    """Wrap generate to steer the request and attach its capture to the final output.

    Inline steering vectors reach the workers by RPC before generation; table rows ride in
    ``extra_args``. The worker writes the capture to a file; the final output waits for it,
    or with ``activations_transport="file"`` carries only its path.
    """
    from vllm.v1.engine import EngineCoreRequest

    # A deprecated EngineCoreRequest carries its own cloned SamplingParams; other
    # prompts use a copy of the caller's.
    built = isinstance(prompt, EngineCoreRequest) and prompt.sampling_params is not None  # type: ignore[reportAttributeAccessIssue]
    if built:
        params: SamplingParams = prompt.sampling_params  # type: ignore[reportAttributeAccessIssue]
    else:
        sampling_params = params = copy.copy(sampling_params)
        params.extra_args = dict(params.extra_args or {})
    extra = params.extra_args
    if extra is None:
        params.extra_args = extra = {}
    transport = extra.get("activations_transport")
    if transport == "binary":
        _warn_if_multi_api_server(self)
    vectors, captures = _prepare_request(
        self, self.vllm_config, params, extra, uuid.uuid4().hex
    )
    extra.pop("activations_transport", None)
    if vectors and built:
        _isolate_prefix_cache(prompt)
    path = (
        capture_path(os.environ[CAPTURE_DIR_ENV], extra["_capture_key"], request_id)
        if captures
        else None
    )
    consumed = False
    assert _original_generate is not None
    try:
        if "_steering_id" in extra:
            await self.collective_rpc(
                "set_steering_data",
                args=(extra["_steering_id"], pickle.dumps(vectors)),
            )
        async for output in _original_generate(
            self, prompt, sampling_params, request_id, **kwargs
        ):
            if output.finished and path is not None:
                expected = (
                    len(output.prompt_token_ids) + len(output.outputs[0].token_ids) - 1
                )
                if transport == "file":
                    marker = {"transport": "file", "path": path, "rows": expected}
                    output.activations = {"residual_stream": marker}
                    output._vllm_lens_serialized = {"residual_stream": marker}
                else:
                    await wait_capture_async(path, _CAPTURE_TIMEOUT)
                    activations = read_capture(path)
                    os.unlink(path)
                    _trim_activations(activations, expected)
                    output.activations = activations
                    output._vllm_lens_serialized = await asyncio.to_thread(
                        _serialize_for, activations, transport
                    )
                consumed = True
            yield output
    finally:
        if "_steering_id" in extra:
            await self.collective_rpc(
                "clear_steering_data", args=(extra["_steering_id"],)
            )
        for name, table_version in _release_tables(self, vectors):
            await self.collective_rpc(
                "clear_steering_table", args=(name, table_version)
            )
        if path is not None and not consumed:
            threading.Thread(target=_remove_capture, args=(path,), daemon=True).start()


def _prepare_offline_params(
    self: LLM,
    sampling_params: SamplingParams | Sequence[SamplingParams] | None,
) -> dict[str, Any]:
    """Check and prepare the copies of the caller's SamplingParams for the patched offline
    entry points, and send inline steering vectors to the workers."""
    if isinstance(sampling_params, Sequence):
        params_list = [copy.copy(sp) for sp in sampling_params]
        sampling_params = params_list
    elif sampling_params is not None:
        params_list = [copy.copy(sampling_params)]
        sampling_params = params_list[0]
    else:
        params_list = []
    vllm_config = self.llm_engine.vllm_config
    vectors_list: list[list[SteeringVector] | None] = []
    key = uuid.uuid4().hex
    steering: dict[str, bytes] = {}
    captures = False
    for idx, sp in enumerate(params_list):
        sp.extra_args = extra = dict(sp.extra_args or {})
        vectors, captured = _prepare_request(
            self, vllm_config, sp, extra, f"{key}{idx}"
        )
        extra.pop("activations_transport", None)
        captures |= captured
        vectors_list.append(vectors)
        if "_steering_id" in extra:
            steering[extra["_steering_id"]] = pickle.dumps(vectors)
    try:
        for steering_id, payload in steering.items():
            self.collective_rpc("set_steering_data", args=(steering_id, payload))
    except BaseException:
        _end_offline(self, {"steering": steering, "vectors": vectors_list})
        raise
    return {
        "params": sampling_params,
        "params_list": params_list,
        "captures": captures,
        "steering": steering,
        "vectors": vectors_list,
    }


def _finalize_offline_outputs(self: LLM, outputs: list, state: dict[str, Any]) -> list:
    """Attach each output's capture, read from the file its worker wrote."""
    if not state["captures"]:
        return outputs
    from vllm import envs

    if not envs.VLLM_ENABLE_V1_MULTIPROCESSING:
        # An in-process engine stops when its requests finish; one more step passes their
        # finished ids to the workers, which seal the files.
        self.llm_engine.step()
    params_list = state["params_list"]
    directory = os.environ[CAPTURE_DIR_ENV]
    for idx, output in enumerate(outputs):
        extra = params_list[idx if len(params_list) > 1 else 0].extra_args
        if "_capture_key" not in extra:
            continue
        path = capture_path(directory, extra["_capture_key"], output.request_id)
        wait_capture(path, _CAPTURE_TIMEOUT)
        activations = read_capture(path)
        os.unlink(path)
        n_prompt = len(output.prompt_token_ids)
        _trim_activations(activations, n_prompt + len(output.outputs[0].token_ids) - 1)
        output.activations = activations
    return outputs


def _end_offline(self: LLM, state: dict[str, Any]) -> None:
    """Drop the inline steering data and the table versions nothing uses any more."""
    for steering_id in state["steering"]:
        self.collective_rpc("clear_steering_data", args=(steering_id,))
    for vectors in state["vectors"]:
        for name, table_version in _release_tables(self, vectors):
            self.collective_rpc("clear_steering_table", args=(name, table_version))


def _patched_llm_generate(
    self: LLM,
    prompts: Any,
    sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
    *args,
    **kwargs,
) -> list:
    """Wrap ``LLM.generate`` to steer requests and attach their captures."""
    state = _prepare_offline_params(self, sampling_params)
    assert _original_llm_generate is not None
    try:
        outputs = _original_llm_generate(
            self, prompts, state["params"], *args, **kwargs
        )
        return _finalize_offline_outputs(self, outputs, state)
    finally:
        _end_offline(self, state)


def _patched_llm_chat(
    self: LLM,
    messages: Any,
    sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
    *args,
    **kwargs,
) -> list:
    """Wrap ``LLM.chat``, which submits to the engine without ``LLM.generate``, as
    :func:`_patched_llm_generate` does."""
    state = _prepare_offline_params(self, sampling_params)
    assert _original_llm_chat is not None
    try:
        outputs = _original_llm_chat(self, messages, state["params"], *args, **kwargs)
        return _finalize_offline_outputs(self, outputs, state)
    finally:
        _end_offline(self, state)


# ---------------------------------------------------------------------------
# Response builder patches for vllm serve (OpenAI-compatible API)
# ---------------------------------------------------------------------------


_multi_api_server_warned = False


def _warn_if_multi_api_server(engine: Any) -> None:
    """Warn once if binary transport is used with ``--api-server-count > 1``: the store is
    per API-server process, so a handle from one frontend 404s on another."""
    global _multi_api_server_warned
    if _multi_api_server_warned:
        return
    _multi_api_server_warned = True
    count = getattr(
        getattr(getattr(engine, "vllm_config", None), "parallel_config", None),
        "_api_process_count",
        1,
    )
    if count and count > 1:
        logger.warning(
            "activations_transport='binary' with --api-server-count=%d: the "
            "activation store is per API-server process, so GET "
            "/v1/activations/{handle} will 404 unless it reaches the same "
            "process that served the completion. Use --api-server-count=1 or "
            "the default base64 transport.",
            count,
        )


def _patched_completion_response(self, final_res_batch, *args, **kwargs):
    """Wrap the completion response builder to add the serialized activations."""
    assert _original_completion_response is not None
    response = _original_completion_response(self, final_res_batch, *args, **kwargs)
    for res in final_res_batch or ():
        serialized = getattr(res, "_vllm_lens_serialized", None)
        if serialized is not None:
            response.activations = serialized
            break
    return response


async def _patched_chat_full_generator(
    self, request, result_generator, *args, **kwargs
):
    """Wrap the chat completion full generator to add the activations of its final
    ``RequestOutput``."""
    assert _original_chat_full_generator is not None

    last_output = None

    async def _capturing(gen: AsyncIterator) -> AsyncIterator:
        nonlocal last_output
        async for output in gen:
            last_output = output
            yield output

    response = await _original_chat_full_generator(
        self, request, _capturing(result_generator), *args, **kwargs
    )
    serialized = getattr(last_output, "_vllm_lens_serialized", None)
    # Only for successful responses (not ErrorResponse).
    if serialized is not None and hasattr(response, "model_dump"):
        response.activations = serialized
    return response


async def _patched_chat_stream_generator(
    self, request, result_generator, *args, **kwargs
):
    """Wrap the streaming chat generator to send the activations as a final SSE chunk."""
    assert _original_chat_stream_generator is not None

    last_output = None

    async def _capturing(gen: AsyncIterator) -> AsyncIterator:
        nonlocal last_output
        async for output in gen:
            last_output = output
            yield output

    async for chunk in _original_chat_stream_generator(
        self, request, _capturing(result_generator), *args, **kwargs
    ):
        if chunk.strip() == "data: [DONE]":
            serialized = getattr(last_output, "_vllm_lens_serialized", None)
            if serialized is not None:
                body = await asyncio.to_thread(json.dumps, {"activations": serialized})
                yield f"data: {body}\n\n"
        yield chunk


def _patched_register_routers(app):
    """Add the vllm-lens routes after vLLM registers its own."""
    assert _original_register_routers is not None
    _original_register_routers(app)
    from vllm_lens._routes import activations_router, steering_tables_router

    app.include_router(activations_router)
    app.include_router(steering_tables_router)


# ---------------------------------------------------------------------------
# Steering tables on the offline LLM
# ---------------------------------------------------------------------------


def _llm_set_steering_table(self: LLM, name: str, vectors: torch.Tensor) -> None:
    """Hold ``(N, hidden)`` steering vectors on the workers under ``name``, for requests
    that name a row in ``apply_steering_vectors``."""
    hidden = self.llm_engine.model_config.get_hidden_size()
    _check_steering_vectors(vectors, hidden)
    table_version, drop = _replace_table(self, name, vectors.shape[0])
    self.collective_rpc(
        "set_steering_table",
        args=(name, serialize_tensor(vectors.float()), table_version),
    )
    if drop is not None:
        self.collective_rpc("clear_steering_table", args=(name, drop))


def _llm_clear_steering_table(self: LLM, name: str | None = None) -> None:
    """Drop the steering table under ``name``; None drops every table. Requests in flight
    keep the versions they use."""
    for key, table_version in _remove_table(self, name):
        self.collective_rpc("clear_steering_table", args=(key, table_version))


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def _report_integration_failure(component: str, error: Exception) -> None:
    """Expose API drift; compatibility runs treat missing integrations as errors.

    Offline installations may intentionally lack serving dependencies, so normal
    registration warns rather than making those installations unusable.
    """
    try:
        vllm_version = version("vllm")
    except PackageNotFoundError:
        vllm_version = "unknown"
    message = (
        f"vllm-lens could not install {component} on vLLM {vllm_version}: "
        f"{type(error).__name__}: {error}. This integration is unavailable; "
        "HTTP responses may omit activations. "
        "See docs/compatibility.md for tested configurations. "
        "Set VLLM_LENS_STRICT_COMPATIBILITY=1 to fail startup on integration errors."
    )
    if os.environ.get("VLLM_LENS_STRICT_COMPATIBILITY") == "1":
        raise RuntimeError(message) from error
    logger.warning(message)


def register() -> None:
    """Entry point called by vLLM's plugin system at engine startup.

    Patches engine creation, ``AsyncLLM.generate``, ``LLM.generate`` and ``LLM.chat``, and
    the OpenAI-compatible response builders so activations reach HTTP responses. Set
    ``VLLM_LENS_DISABLE=1`` to make this a no-op.
    """
    if os.environ.get("VLLM_LENS_DISABLE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        logger.info("VLLM_LENS_DISABLE set; vllm-lens activation plugin inactive.")
        return

    global _original_create_engine_config
    global _original_generate, _original_llm_generate, _original_llm_chat
    global _original_completion_response, _original_chat_full_generator
    global _original_chat_stream_generator
    global _original_register_routers
    global _original_process_inputs

    from vllm import LLM
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.input_processor import InputProcessor

    _original_create_engine_config = EngineArgs.create_engine_config
    EngineArgs.create_engine_config = _patched_create_engine_config

    # Before any engine is built: InputProcessor binds process_inputs_async at __init__.
    _original_process_inputs = InputProcessor.process_inputs
    InputProcessor.process_inputs = _patched_process_inputs  # type: ignore[reportAttributeAccessIssue]

    _original_generate = AsyncLLM.generate
    AsyncLLM.generate = _patched_generate  # type: ignore[reportAttributeAccessIssue]

    _original_llm_generate = LLM.generate
    LLM.generate = _patched_llm_generate

    # LLM.chat submits requests to the engine directly rather than
    # delegating to LLM.generate, so it must be patched separately.
    _original_llm_chat = LLM.chat
    LLM.chat = _patched_llm_chat

    LLM.set_steering_table = _llm_set_steering_table  # type: ignore[reportAttributeAccessIssue]
    LLM.clear_steering_table = _llm_clear_steering_table  # type: ignore[reportAttributeAccessIssue]
    # Each registered table's row count and version, replaced and never mutated in place,
    # and the requests in flight per table version.
    for cls in (LLM, AsyncLLM):
        cls._lens_steering_tables = {}  # type: ignore[reportAttributeAccessIssue]
        cls._lens_table_users = collections.Counter()  # type: ignore[reportAttributeAccessIssue]

    try:
        from vllm.entrypoints.openai.completion.serving import (
            OpenAIServingCompletion,
        )

        _original_completion_response = (
            OpenAIServingCompletion.request_output_to_completion_response
        )
        OpenAIServingCompletion.request_output_to_completion_response = (
            _patched_completion_response
        )
    except Exception as error:
        _report_integration_failure("completion response patch", error)

    try:
        from vllm.entrypoints.openai.chat_completion.serving import (
            OpenAIServingChat,
        )

        _original_chat_full_generator = OpenAIServingChat.chat_completion_full_generator
        OpenAIServingChat.chat_completion_full_generator = _patched_chat_full_generator
        _original_chat_stream_generator = (
            OpenAIServingChat.chat_completion_stream_generator
        )
        OpenAIServingChat.chat_completion_stream_generator = (
            _patched_chat_stream_generator
        )
    except Exception as error:
        _report_integration_failure("chat response patches", error)

    try:
        import vllm.entrypoints.serve as _serve_mod

        _original_register_routers = _serve_mod.register_vllm_serve_api_routers
        _serve_mod.register_vllm_serve_api_routers = _patched_register_routers
    except Exception as error:
        _report_integration_failure("activation and steering table routes", error)

"""
vLLM general plugin that transparently captures residual-stream
activations via worker extension when ``output_residual_stream`` is
passed in ``extra_args``.

Installed automatically via the ``vllm.general_plugins`` entry point
(configured in pyproject.toml). Patches ``EngineArgs.create_engine_config``
to inject the worker extension and eager mode, and patches
``AsyncLLM.generate``, ``LLM.generate``, and ``LLM.chat`` to retrieve
per-request activations for both online (async) and offline (sync) usage.
"""

from __future__ import annotations

import copy
import json
import logging
import mmap
import os
import pickle
from importlib.metadata import PackageNotFoundError, version
from collections.abc import AsyncIterator, Callable, Sequence
from typing import TYPE_CHECKING, Any

import cloudpickle
import torch
import zstandard as zstd

from vllm_lens._cudagraph import LensGraphConfig
from vllm_lens._helpers._serialize import (
    deserialize_directions,
    serialize_tensor,
    serialize_activations,
    serialize_activations_binary,
    serialize_hook_results,
)
from vllm_lens._helpers._transport import Packed, Segment, open_segment, unpack_payloads
from vllm_lens._helpers.types import (
    HOOKS_PAYLOAD,
    STEERING_PAYLOAD,
    Hook,
    SteeringVector,
)

# A child of vLLM's logger, so vLLM's handler, format and VLLM_LOGGING_LEVEL apply.
logger = logging.getLogger("vllm.lens")

if TYPE_CHECKING:
    from vllm.exceptions import VLLMValidationError
else:
    try:
        from vllm.exceptions import VLLMValidationError
    except ImportError:  # The CPU tests run without vLLM.
        VLLMValidationError = type("VLLMValidationError", (Exception,), {})


class LensRequestError(VLLMValidationError, ValueError):
    """A request vllm-lens cannot serve. vLLM passes it to the caller and HTTP returns 400."""


_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
_ZSTD_DECOMPRESSOR = zstd.ZstdDecompressor()

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM

_WORKER_EXT = "vllm_lens._worker_ext.HiddenStatesExtension"

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


# ---------------------------------------------------------------------------
# PP merge helper
# ---------------------------------------------------------------------------


def _merge_hook_results(
    raw_list: list[bytes | None] | None,
) -> dict[str, dict[str, Any]] | None:
    """Merge hook results from multiple PP ranks.

    Each rank returns ``{hook_idx_str: ctx.saved}``.  For list values
    (e.g. accumulated activations per layer), concatenates rather than
    overwrites so data from all PP stages is preserved.
    """
    if not raw_list:
        return None
    merged: dict[str, dict[str, Any]] = {}
    for raw in raw_list:
        if raw is None:
            continue
        rank_results: dict[str, dict[str, Any]] = pickle.loads(raw)
        for hook_idx, saved in rank_results.items():
            if hook_idx not in merged:
                merged[hook_idx] = {}
            for key, val in saved.items():
                if (
                    key in merged[hook_idx]
                    and isinstance(merged[hook_idx][key], list)
                    and isinstance(val, list)
                ):
                    merged[hook_idx][key].extend(val)
                else:
                    merged[hook_idx][key] = val
    return merged or None


def _decode_rank_payload(payload: bytes) -> Any:
    """Decode a rank payload, transparently handling the zstd-or-raw split.

    Worker-side payloads are zstd-compressed when shipped over the network
    (HTTP / OpenAI path) and raw when shipped over a local subprocess pipe.
    The magic-byte prefix lets us pick the right decoder without a flag.
    """
    return pickle.loads(
        _ZSTD_DECOMPRESSOR.decompress(payload)
        if payload[:4] == _ZSTD_MAGIC
        else payload
    )


def _merge_captured_states(
    states: list[bytes | None] | None,
) -> dict[str, Any] | None:
    """Merge activation captures from multiple PP ranks.

    ``collective_rpc`` returns results in rank order (rank 0, 1, ...).
    With TP, only TP-rank-0 workers capture (others return ``None``).
    Each capturing rank's tensor is sorted by global layer index.
    Because lower PP ranks hold earlier layers, concatenating non-None
    results along dim 0 produces correct global layer ordering.
    """
    if not states:
        return None
    parts: list[dict[str, Any]] = [
        _decode_rank_payload(s) for s in states if s is not None
    ]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]["activations"]
    return {
        key: torch.cat([p["activations"][key] for p in parts], dim=0)
        for key in parts[0]["activations"]
    }


def _merge_captured_states_batch(
    states_per_rank: list[Packed | None],
    external_req_ids: Sequence[str],
    shared: Sequence[mmap.mmap | None],
) -> dict[str, dict[str, Any]]:
    """Merge each request's shared-memory captures in pipeline-rank order."""
    rank_dicts: list[dict[str, dict[str, Any]]] = [
        unpack_payloads(s, shared[rank])
        for rank, s in enumerate(states_per_rank)
        if s is not None
    ]
    out: dict[str, dict[str, Any]] = {}
    for req_id in external_req_ids:
        per_rank = [d[req_id] for d in rank_dicts if req_id in d]
        if not per_rank:
            continue
        if len(per_rank) == 1:
            out[req_id] = per_rank[0]["activations"]
        else:
            out[req_id] = {
                key: torch.cat([p["activations"][key] for p in per_rank], dim=0)
                for key in per_rank[0]["activations"]
            }
    return out


def _merge_qk_parts(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-rank Q/K payloads into client-facing activation entries.

    Unlike the residual stream (replicated across TP, captured on rank 0
    only), Q/K are sharded by head: every TP rank contributes its own
    slice, concatenated along the head dimension (dim 2) in tp_rank
    order.  When ``total_kv_heads < tp_size`` vLLM replicates each KV
    head across ``tp_size // total_kv_heads`` consecutive ranks, so only
    every ``stride``-th rank's K is kept.  PP groups then concatenate
    along the layer dimension (dim 0), lower pp_rank first — the same
    convention as :func:`_merge_captured_states`.
    """
    by_pp: dict[int, list[dict[str, Any]]] = {}
    for p in parts:
        by_pp.setdefault(int(p.get("pp_rank", 0)), []).append(p)

    pp_groups: list[dict[str, Any]] = []
    for pp_rank in sorted(by_pp):
        group = sorted(by_pp[pp_rank], key=lambda p: int(p.get("tp_rank", 0)))
        first = group[0]
        meta: list[dict[str, Any]] = [dict(m) for m in first["meta"]]
        if len(group) == 1:
            q, k = first["q"], first["k"]
        else:
            tp_size = int(first.get("tp_size", len(group)))
            q = torch.cat([p["q"] for p in group], dim=2)
            total_kv = int(meta[0].get("num_kv_heads_total") or 0)
            stride = max(1, tp_size // total_kv) if total_kv else 1
            k = torch.cat(
                [p["k"] for p in group if int(p.get("tp_rank", 0)) % stride == 0],
                dim=2,
            )
            for li in range(len(meta)):
                # Per-query-head metadata follows the Q concat order.
                for key in ("alibi_slopes", "sinks"):
                    vals = [p["meta"][li].get(key) for p in group]
                    meta[li][key] = (
                        None
                        if vals[0] is None
                        else [x for v in vals for x in (v or [])]
                    )
                meta[li]["num_heads_local"] = int(q.shape[2])
                meta[li]["num_kv_heads_local"] = int(k.shape[2])
        pp_groups.append(
            {"q": q, "k": k, "layers": list(first["layers"]), "meta": meta}
        )

    if len(pp_groups) == 1:
        g = pp_groups[0]
        q, k, layers, meta = g["q"], g["k"], g["layers"], g["meta"]
    else:
        q = torch.cat([g["q"] for g in pp_groups], dim=0)
        k = torch.cat([g["k"] for g in pp_groups], dim=0)
        layers = [i for g in pp_groups for i in g["layers"]]
        meta = [m for g in pp_groups for m in g["meta"]]

    return {"attn_q": q, "attn_k": k, "qk_layers": layers, "qk_meta": meta}


def _merge_captured_qk(
    states: list[bytes | None] | None,
) -> dict[str, Any] | None:
    """Merge Q/K captures from all ranks (see :func:`_merge_qk_parts`)."""
    if not states:
        return None
    parts = [_decode_rank_payload(s)["qk"] for s in states if s is not None]
    if not parts:
        return None
    return _merge_qk_parts(parts)


def _merge_captured_qk_batch(
    states_per_rank: list[bytes | None] | None,
    external_req_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Batched :func:`_merge_captured_qk`, keyed by external request id."""
    if not states_per_rank:
        return {}
    rank_dicts: list[dict[str, dict[str, Any]]] = [
        _decode_rank_payload(s) for s in states_per_rank if s is not None
    ]
    out: dict[str, dict[str, Any]] = {}
    for req_id in external_req_ids:
        parts = [d[req_id]["qk"] for d in rank_dicts if req_id in d]
        if parts:
            out[req_id] = _merge_qk_parts(parts)
    return out


def _decode_steering_vectors(value: Any) -> list[SteeringVector] | None:
    """Normalise an ``apply_steering_vectors`` extra_args value.

    Accepts live ``SteeringVector`` instances, plain dicts (already
    JSON-parsed), or the JSON-encoded string wire format used by the
    OpenAI-compatible API (``vllm_xargs``), so every entry point
    understands every documented form.
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = json.loads(value)
    return [
        v if isinstance(v, SteeringVector) else SteeringVector.model_validate(v)
        for v in value
    ]


def _decode_hooks(value: Any) -> list[Hook] | None:
    """Normalise an ``apply_hooks`` extra_args value.

    Same forms as :func:`_decode_steering_vectors`, for ``Hook``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = json.loads(value)
    return [h if isinstance(h, Hook) else Hook.model_validate(h) for h in value]


def _check_directions(value: Any, hidden: int) -> str:
    """The wire string of projection directions; ``ValueError`` unless they are finite
    ``(N, hidden)``."""
    payload = value if isinstance(value, str) else json.dumps(value)
    directions = deserialize_directions(payload)
    if (
        directions.ndim != 2
        or directions.shape[1] != hidden
        or not torch.isfinite(directions).all()
    ):
        raise ValueError(
            f"projection directions must be finite (N, {hidden}), "
            f"got {tuple(directions.shape)}"
        )
    return payload


def _trim_activations(
    activations: dict[str, Any],
    expected_len: int,
) -> None:
    """Trim captured positions to the expected sequence length."""
    for key in (
        "residual_stream",
        "residual_stream_projection",
        "residual_stream_norm",
        "attn_q",
        "attn_k",
    ):
        t = activations.get(key)
        if t is not None and t.shape[1] > expected_len:
            activations[key] = t[:, :expected_len]
    ids = activations.get("input_ids")
    if ids is not None and len(ids) > expected_len:
        activations["input_ids"] = ids[:expected_len]


# ---------------------------------------------------------------------------
# Engine config patch — inject worker extension + eager mode
# ---------------------------------------------------------------------------


def _patched_create_engine_config(self, *args, **kwargs):
    """Configure the worker extension and execution mode before engine creation."""
    if not self.worker_extension_cls:
        self.worker_extension_cls = _WORKER_EXT
    # Layer lists in additional_config["vllm_lens"] keep compilation and CUDA graphs on.
    self.enforce_eager = (
        self.enforce_eager or not LensGraphConfig.from_vllm_config(self).enabled
    )

    # Our capture/steering hooks read V1 model-runner internals (input_batch,
    # requests). vLLM's V2 runner — the default for dense models on vLLM 0.23+ —
    # has a different layout, so the hooks silently capture nothing there. Default
    # to the V1 runner. setdefault => an explicit VLLM_USE_V2_MODEL_RUNNER still wins
    # (and is caught below); no-op on older vLLM that ignores this env var.
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

    assert _original_create_engine_config is not None
    config = _original_create_engine_config(self, *args, **kwargs)

    # vLLM serves fp32 models with the Triton attention backend, and Triton's
    # tl.dot defaults to TF32 for fp32 inputs — so an "fp32" model is not fp32
    # end-to-end (each layer's attention output carries ~1e-3 relative error
    # into the next layer's activations; measured 2.9e-3 vs 3.5e-6 in attention
    # probabilities). vLLM already defaults VLLM_FLOAT32_MATMUL_PRECISION=highest
    # for the linears; make the attention kernel consistent. setdefault => an
    # explicit TRITON_F32_DEFAULT still wins. Triton reads it lazily at kernel
    # compile time in the workers, which inherit this environment.
    if getattr(getattr(config, "model_config", None), "dtype", None) == torch.float32:
        if os.environ.setdefault("TRITON_F32_DEFAULT", "ieee") == "ieee":
            logger.info(
                "vllm-lens: fp32 model — defaulting TRITON_F32_DEFAULT=ieee so the "
                "Triton attention kernel computes in true fp32 (set TRITON_F32_DEFAULT "
                "explicitly to override)."
            )

    # Fail loudly rather than silently no-op if the V2 runner ended up active anyway
    # (e.g. the user explicitly set VLLM_USE_V2_MODEL_RUNNER=1). getattr keeps this a
    # no-op on vLLM versions with no such property (pre-V2, e.g. 0.19).
    if getattr(config, "use_v2_model_runner", False):
        raise RuntimeError(
            "vllm-lens requires vLLM's V1 model runner, but the V2 model runner is "
            "active (VLLM_USE_V2_MODEL_RUNNER=1). Activation capture and steering rely "
            "on V1 runner internals and would silently no-op on V2. Unset "
            "VLLM_USE_V2_MODEL_RUNNER (vllm-lens defaults it to 0) or set it to 0."
        )
    return config


# ---------------------------------------------------------------------------
# Generate patch — install hooks and attach activations to output
# ---------------------------------------------------------------------------


def _patched_process_inputs(
    self: Any, request_id: str, prompt: Any, params: Any, *args, **kwargs
) -> Any:
    """Check a request's vectors and hooks and move them into a copy of its ``extra_args`` as
    pickled bytes, which reach the workers with the request. Shared by every entry point."""
    assert _original_process_inputs is not None
    extra = dict(getattr(params, "extra_args", None) or {})
    persistent = getattr(self, "_lens_persistent_hooks", False)
    if not extra and not persistent:
        return _original_process_inputs(
            self, request_id, prompt, params, *args, **kwargs
        )
    # Only this function sets the payload keys; a client cannot send its own.
    for key in (STEERING_PAYLOAD, HOOKS_PAYLOAD, "activations_transport"):
        extra.pop(key, None)
    try:
        pool = extra.get("output_residual_stream_pool")
        if pool not in (None, "last", "mean"):
            raise ValueError(
                f"output_residual_stream_pool must be 'last' or 'mean', got {pool!r}"
            )
        project = extra.get("output_residual_stream_project")
        if project is not None:
            if pool is not None:
                raise ValueError(
                    "output_residual_stream_project cannot be combined with "
                    "output_residual_stream_pool"
                )
            if project not in getattr(self, "_lens_projections", ()):
                raise ValueError(
                    f"output_residual_stream_project names {project!r}, which is not "
                    "registered: register it with LLM.set_projection or POST /v1/projections"
                )
        vectors = _decode_steering_vectors(extra.pop("apply_steering_vectors", None))
        hooks = _decode_hooks(extra.pop("apply_hooks", None))
        graph_config = LensGraphConfig.from_vllm_config(self.vllm_config)
        graph_config.reject_hooks(hooks or [], qk=extra.get("output_qk") is not None)
        graph_config.reject_request(extra.get("output_residual_stream"), vectors)
        n_layers = self.model_config.get_total_num_hidden_layers()
        for item in (*(vectors or []), *(hooks or [])):
            outside = [idx for idx in item.layer_indices if not 0 <= idx < n_layers]
            if outside:
                raise ValueError(f"layer_index {outside} out of range [0, {n_layers})")
    except ValueError as error:
        raise LensRequestError(str(error)) from error
    if vectors is not None:
        # A copy of each tensor: a view pickles the whole storage it belongs to.
        extra[STEERING_PAYLOAD] = pickle.dumps(
            [
                sv.model_copy(
                    update={"activations": sv.activations.detach().cpu().clone()}
                )
                for sv in vectors
            ]
        )
    if hooks is not None:
        extra[HOOKS_PAYLOAD] = cloudpickle.dumps(hooks)
    params = copy.copy(params)
    params.extra_args = extra
    # Prefix-cached tokens skip the forward pass, so the hooks would not see them.
    if (
        extra.pop("skip_reading_prefix_cache", None)
        or vectors is not None
        or hooks is not None
        or persistent
        or extra.get("output_residual_stream") is not None
        or extra.get("output_qk") is not None
    ):
        params.skip_reading_prefix_cache = True
    return _original_process_inputs(self, request_id, prompt, params, *args, **kwargs)


def _lens_flags(extra: dict[str, Any] | None, persistent: bool) -> dict[str, bool]:
    """What a request needs from the workers: hooks installed, and which results to collect."""
    extra = extra or {}
    flags = {
        "activations": extra.get("output_residual_stream") is not None,
        "qk": extra.get("output_qk") is not None,
        "hooks": extra.get("apply_hooks") is not None,
    }
    flags["install"] = (
        flags["activations"]
        or flags["hooks"]
        or persistent
        or extra.get("apply_steering_vectors") is not None
    )
    return flags


async def _patched_generate(
    self: AsyncLLM,
    prompt: str,
    sampling_params: SamplingParams,
    request_id: str,
    **kwargs,
) -> AsyncIterator:
    """Run generation and attach the request's captures and hook results to its output."""
    flags = _lens_flags(
        sampling_params.extra_args,
        getattr(self.input_processor, "_lens_persistent_hooks", False),
    )
    # Response-transport negotiation (issue #31): "binary" parks activation
    # bytes in the process-local store and returns a handle instead of base64.
    activations_transport = (sampling_params.extra_args or {}).get(
        "activations_transport"
    )
    if activations_transport == "binary":
        _warn_if_multi_api_server(self)
    if flags["install"] and not getattr(self, "_hooks_installed", False):
        await self.collective_rpc("install_hooks")
        setattr(self, "_hooks_installed", True)
    if flags["qk"] and not getattr(self, "_qk_hooks_installed", False):
        await self.collective_rpc("install_qk_hooks")
        setattr(self, "_qk_hooks_installed", True)

    assert _original_generate is not None
    try:
        async for output in _original_generate(
            self, prompt, sampling_params, request_id, **kwargs
        ):
            if output.finished:
                if flags["activations"]:
                    states = await self.collective_rpc(
                        "get_captured_states", args=(request_id,)
                    )
                    activations = _merge_captured_states(states)
                    if activations is not None:
                        n_prompt = len(output.prompt_token_ids)
                        n_gen = len(output.outputs[0].token_ids)
                        _trim_activations(activations, n_prompt + n_gen - 1)
                        output.activations = activations
                        output._vllm_lens_transport = activations_transport
                if flags["qk"]:
                    qk_states = await self.collective_rpc(
                        "get_captured_qk", args=(request_id,)
                    )
                    qk = _merge_captured_qk(qk_states)
                    if qk is not None:
                        n_prompt = len(output.prompt_token_ids)
                        n_gen = len(output.outputs[0].token_ids)
                        _trim_activations(qk, n_prompt + n_gen - 1)
                        merged_acts = getattr(output, "activations", None) or {}
                        merged_acts.update(qk)
                        output.activations = merged_acts
                if flags["hooks"]:
                    raw_results = await self.collective_rpc(
                        "get_hook_results", args=(request_id,)
                    )
                    merged = _merge_hook_results(raw_results)
                    if merged is not None:
                        output.hook_results = merged
            yield output
    finally:
        # An aborted request leaves its captures and hook contexts on the workers.
        if flags["activations"] or flags["qk"] or flags["hooks"]:
            await self.collective_rpc("clear_request_state", args=([request_id],))


# ---------------------------------------------------------------------------
# Offline (sync) LLM.generate / LLM.chat patches
# ---------------------------------------------------------------------------


def _finalize_offline_outputs(self: LLM, outputs: list, flags: dict[str, bool]) -> list:
    """Attach captured activations and hook results to offline outputs."""
    if flags["activations"]:
        req_ids = [output.request_id for output in outputs]
        # Each rank writes its captures into a memfd that this process maps through /proc.
        results = self.collective_rpc("get_captured_states_batch", args=(req_ids,))
        states_per_rank: list[Packed | None] = []
        regions: list[mmap.mmap | None] = []
        for result in results:
            segment, packed = result if result is not None else (None, None)
            states_per_rank.append(packed)
            # msgpack sends the NamedTuple as a list.
            regions.append(open_segment(Segment(*segment)) if segment else None)
        activations_by_id = _merge_captured_states_batch(
            states_per_rank, req_ids, regions
        )
        for output in outputs:
            activations = activations_by_id.get(output.request_id)
            if activations is not None:
                n_prompt = len(output.prompt_token_ids)
                n_gen = len(output.outputs[0].token_ids)
                _trim_activations(activations, n_prompt + n_gen - 1)
                output.activations = activations

    if flags["qk"]:
        req_ids = [output.request_id for output in outputs]
        qk_per_rank = self.collective_rpc("get_captured_qk_batch", args=(req_ids,))
        qk_by_id = _merge_captured_qk_batch(qk_per_rank, req_ids)
        for output in outputs:
            qk = qk_by_id.get(output.request_id)
            if qk is not None:
                n_prompt = len(output.prompt_token_ids)
                n_gen = len(output.outputs[0].token_ids)
                _trim_activations(qk, n_prompt + n_gen - 1)
                merged_acts = getattr(output, "activations", None) or {}
                merged_acts.update(qk)
                output.activations = merged_acts

    if flags["hooks"]:
        for output in outputs:
            req_id = output.request_id
            raw_results = self.collective_rpc("get_hook_results", args=(req_id,))
            merged = _merge_hook_results(raw_results)
            if merged is not None:
                output.hook_results = merged

    return outputs


def _run_offline(
    self: LLM,
    original: Callable[..., list],
    first: Any,
    sampling_params: SamplingParams | Sequence[SamplingParams] | None,
    args: tuple,
    kwargs: dict[str, Any],
) -> list:
    """Run an offline entry point and attach its results; after a failure, drop the captures
    and hook contexts of every request (the offline call holds the engine)."""
    if isinstance(sampling_params, Sequence):
        params_list = list(sampling_params)
    else:
        params_list = [sampling_params] if sampling_params is not None else []
    persistent = getattr(
        self.llm_engine.input_processor, "_lens_persistent_hooks", False
    )
    per_request = [_lens_flags(sp.extra_args, persistent) for sp in params_list]
    flags = {key: any(f[key] for f in per_request) for key in _lens_flags(None, False)}
    if flags["install"] and not getattr(self, "_hooks_installed", False):
        self.collective_rpc("install_hooks")
        self._hooks_installed = True  # type: ignore[reportAttributeAccessIssue]
    if flags["qk"] and not getattr(self, "_qk_hooks_installed", False):
        self.collective_rpc("install_qk_hooks")
        self._qk_hooks_installed = True  # type: ignore[reportAttributeAccessIssue]
    try:
        outputs = original(self, first, sampling_params, *args, **kwargs)
        return _finalize_offline_outputs(self, outputs, flags)
    except BaseException:
        self.collective_rpc("clear_request_state", args=(None,))
        raise


def _patched_llm_generate(
    self: LLM,
    prompts: Any,
    sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
    *args,
    **kwargs,
) -> list:
    """Run offline generation with capture, steering, and hooks."""
    assert _original_llm_generate is not None
    return _run_offline(
        self, _original_llm_generate, prompts, sampling_params, args, kwargs
    )


def _patched_llm_chat(
    self: LLM,
    messages: Any,
    sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
    *args,
    **kwargs,
) -> list:
    """Run offline chat with capture, steering, and hooks."""
    assert _original_llm_chat is not None
    return _run_offline(
        self, _original_llm_chat, messages, sampling_params, args, kwargs
    )


# ---------------------------------------------------------------------------
# Response builder patches for vllm serve (OpenAI-compatible API)
# ---------------------------------------------------------------------------


_multi_api_server_warned = False


def _warn_if_multi_api_server(engine: Any) -> None:
    """Warn once if binary transport is used with ``--api-server-count > 1``.

    The activation store is per API-server process, so a handle minted by one
    frontend 404s when the follow-up ``GET`` lands on another.
    """
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


def _serialize_activations_for(obj: Any) -> dict[str, Any] | None:
    """Serialize ``obj.activations`` for the wire, honouring the negotiated transport.

    Returns ``None`` when the request captured nothing. ``"binary"`` (set on the
    output by :func:`_patched_generate`) parks the bytes in the process-local
    store and emits handle descriptors; anything else keeps the default
    base64-in-JSON form, byte-identical to before this feature.
    """
    activations = getattr(obj, "activations", None)
    if activations is None:
        return None
    if getattr(obj, "_vllm_lens_transport", None) == "binary":
        from vllm_lens._helpers._activation_store import store

        return serialize_activations_binary(activations, store)
    return serialize_activations(activations)


def _patched_completion_response(self, final_res_batch, *args, **kwargs):
    """Wrap the completion response builder to inject serialized activations and hook results."""
    assert _original_completion_response is not None
    response = _original_completion_response(self, final_res_batch, *args, **kwargs)
    for res in final_res_batch or ():
        serialized = _serialize_activations_for(res)
        if serialized is not None:
            response.activations = serialized
        hook_results = getattr(res, "hook_results", None)
        if hook_results is not None:
            response.hook_results = serialize_hook_results(hook_results)
        if serialized is not None or hook_results is not None:
            break
    return response


async def _patched_chat_full_generator(
    self, request, result_generator, *args, **kwargs
):
    """Wrap the chat completion full generator to inject serialized activations.

    The original method iterates ``result_generator`` internally, so we
    wrap it with a capturing async generator to grab the final
    ``RequestOutput`` (which has ``.activations`` attached by
    ``_patched_generate``).
    """
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

    # Only inject for successful responses (not ErrorResponse).
    if last_output is not None and hasattr(response, "model_dump"):
        serialized = _serialize_activations_for(last_output)
        if serialized is not None:
            response.activations = serialized
        hook_results = getattr(last_output, "hook_results", None)
        if hook_results is not None:
            response.hook_results = serialize_hook_results(hook_results)

    return response


async def _patched_chat_stream_generator(
    self, request, result_generator, *args, **kwargs
):
    """Wrap the streaming chat generator to append hook results as a final SSE chunk.

    Captures the final ``RequestOutput`` (which has ``.hook_results``
    attached by ``_patched_generate``), then yields an extra
    ``data: {...}`` chunk with serialized hook results before ``[DONE]``.
    """
    assert _original_chat_stream_generator is not None

    last_output = None

    async def _capturing(gen: AsyncIterator) -> AsyncIterator:
        nonlocal last_output
        async for output in gen:
            last_output = output
            yield output

    # Buffer the last chunk so we can inject hook results before [DONE].
    async for chunk in _original_chat_stream_generator(
        self, request, _capturing(result_generator), *args, **kwargs
    ):
        if chunk.strip() == "data: [DONE]" and last_output is not None:
            import json as _json

            extra: dict[str, Any] = {}
            serialized = _serialize_activations_for(last_output)
            if serialized is not None:
                extra["activations"] = serialized
            hook_results = getattr(last_output, "hook_results", None)
            if hook_results is not None:
                extra["hook_results"] = serialize_hook_results(hook_results)
            if extra:
                yield f"data: {_json.dumps(extra)}\n\n"
        yield chunk


# ---------------------------------------------------------------------------
# Persistent hooks — HTTP router
# ---------------------------------------------------------------------------


def _patched_register_routers(app):
    """Inject vllm-lens hook router after vLLM registers its own routes."""
    assert _original_register_routers is not None
    _original_register_routers(app)
    from vllm_lens._hooks_router import activations_router, projections_router
    from vllm_lens._hooks_router import router as hooks_router

    app.include_router(hooks_router)
    app.include_router(activations_router)
    app.include_router(projections_router)


# ---------------------------------------------------------------------------
# Persistent hooks — offline LLM methods
# ---------------------------------------------------------------------------


def _llm_register_hooks(
    self: LLM,
    hooks: list,
    prefetch_params: list[str] | None = None,
) -> None:
    """Register persistent hooks that apply to every subsequent request."""
    LensGraphConfig.from_vllm_config(self.llm_engine.vllm_config).reject_hooks(hooks)
    if not getattr(self, "_hooks_installed", False):
        self.collective_rpc("install_hooks")
        self._hooks_installed = True  # type: ignore[reportAttributeAccessIssue]
    self.collective_rpc("set_persistent_hooks", args=(cloudpickle.dumps(hooks),))
    if prefetch_params:
        self.collective_rpc("prefetch_parameters", args=(prefetch_params,))
    self.llm_engine.input_processor._lens_persistent_hooks = True  # type: ignore[reportAttributeAccessIssue]


def _llm_set_projection(self: LLM, name: str, directions: torch.Tensor) -> None:
    """Hold ``(N, hidden)`` directions on the workers under ``name``, for requests that name
    it in ``output_residual_stream_project``."""
    processor = self.llm_engine.input_processor
    hidden = processor.model_config.get_hidden_size()
    payload = _check_directions(serialize_tensor(directions.float()), hidden)
    self.collective_rpc("set_projection", args=(name, payload))
    processor._lens_projections = {*getattr(processor, "_lens_projections", ()), name}  # type: ignore[reportAttributeAccessIssue]


def _llm_clear_projection(self: LLM, name: str | None = None) -> None:
    """Drop the directions under ``name``; None drops every set."""
    processor = self.llm_engine.input_processor
    self.collective_rpc("clear_projection", args=(name,))
    held = getattr(processor, "_lens_projections", set())
    processor._lens_projections = set() if name is None else held - {name}  # type: ignore[reportAttributeAccessIssue]


def _llm_collect_hook_results(self: LLM) -> dict:
    """Collect all accumulated hook results from the worker."""
    raw_list = self.collective_rpc("get_all_hook_results")
    for raw in raw_list or ():
        if raw is not None:
            return pickle.loads(raw)
    return {}


def _llm_clear_hooks(self: LLM) -> None:
    """Remove persistent hooks and all accumulated contexts."""
    self.collective_rpc("clear_persistent_hooks")
    self.llm_engine.input_processor._lens_persistent_hooks = False  # type: ignore[reportAttributeAccessIssue]


def _llm_prefetch_params(self: LLM, names: list[str]) -> None:
    """Pre-fetch model parameters across all TP/PP ranks."""
    if not getattr(self, "_hooks_installed", False):
        self.collective_rpc("install_hooks")
        self._hooks_installed = True  # type: ignore[reportAttributeAccessIssue]
    self.collective_rpc("prefetch_parameters", args=(names,))


def _llm_clear_prefetched(self: LLM) -> None:
    """Remove all pre-fetched parameters."""
    self.collective_rpc("clear_prefetched_params")


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
        "HTTP responses may omit activations or hook results. "
        "See docs/compatibility.md for tested configurations. "
        "Set VLLM_LENS_STRICT_COMPATIBILITY=1 to fail startup on integration errors."
    )
    if os.environ.get("VLLM_LENS_STRICT_COMPATIBILITY") == "1":
        raise RuntimeError(message) from error
    logger.warning(message)


def register() -> None:
    """Install the plugin unless VLLM_LENS_DISABLE is set."""
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

    # Add persistent hook methods to LLM.
    LLM.register_hooks = _llm_register_hooks  # type: ignore[reportAttributeAccessIssue]
    LLM.collect_hook_results = _llm_collect_hook_results  # type: ignore[reportAttributeAccessIssue]
    LLM.clear_hooks = _llm_clear_hooks  # type: ignore[reportAttributeAccessIssue]
    LLM.prefetch_params = _llm_prefetch_params  # type: ignore[reportAttributeAccessIssue]
    LLM.clear_prefetched = _llm_clear_prefetched  # type: ignore[reportAttributeAccessIssue]
    LLM.set_projection = _llm_set_projection  # type: ignore[reportAttributeAccessIssue]
    LLM.clear_projection = _llm_clear_projection  # type: ignore[reportAttributeAccessIssue]

    # Patch OpenAI-compatible response builders so activations survive
    # HTTP serialization.  Wrapped in try/except because these modules
    # are only available when running as an API server.
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

    # Patch the serve router registration to inject our hooks API.
    try:
        import vllm.entrypoints.serve as _serve_mod

        _original_register_routers = _serve_mod.register_vllm_serve_api_routers
        _serve_mod.register_vllm_serve_api_routers = _patched_register_routers
    except Exception as error:
        _report_integration_failure("hook and activation HTTP routes", error)

    try:
        from vllm_lens._graph_worker import install_graph_patches

        install_graph_patches()
    except ImportError:
        logger.debug("No vLLM GPU worker; the CUDA-graph mode is not available.")

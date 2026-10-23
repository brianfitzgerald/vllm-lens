"""
Worker extension that captures residual-stream activations from
configurable layers during transformer forward passes, and optionally
applies steering vectors (activation additions) to modify the residual
stream in-flight.

Uses PyTorch forward hooks on each decoder layer for concurrency-safe,
per-request activation capture and steering.  Each hook checks the
request's ``extra_args["output_residual_stream"]`` to decide whether to
capture, and reads from ``_steering_data`` to apply any steering vectors.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

import cloudpickle
import torch
import zstandard as zstd
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.models.utils import PPMissingLayer

from vllm_lens._helpers._hook_output import _replace_hook_output
from vllm_lens._helpers.types import Hook, HookContext, SteeringVector

if TYPE_CHECKING:
    from jaxtyping import Float, Int
    from vllm.config import ParallelConfig

logger = logging.getLogger("vllm.lens")

_DTYPE_LIST = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int64,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.float64,
]
_DTYPE_TO_IDX_MAP = {d: i for i, d in enumerate(_DTYPE_LIST)}


def _dtype_to_idx(dtype: torch.dtype) -> int:
    return _DTYPE_TO_IDX_MAP.get(dtype, 0)


_ZSTD_COMPRESSOR = zstd.ZstdCompressor(level=1)


_LAYER_PATH_ENV = "VLLM_LENS_LAYER_PATH"


class LayerDiscoveryError(RuntimeError):
    """vllm-lens could not locate the model's decoder layers."""


def _decoder_mixer_types() -> tuple[type, ...]:
    """Registry classes that live inside a decoder block.

    Only these are treated as decoder-layer markers.  Anything else that
    registers in ``static_forward_context`` (``FusedMoE``, multimodal
    encoder attention, ``CacheOnlyAttentionLayer`` KV-cache dummies,
    DeepSeek-V3.2 indexer caches, future unknowns) is ignored rather than
    guessed at, so a new vLLM layer type can't silently masquerade as a
    decoder layer.
    """
    types: list[type] = []
    from vllm.model_executor.layers.attention import Attention, MLAAttention

    types += [Attention, MLAAttention]
    try:
        from vllm.model_executor.layers.mamba.abstract import MambaBase

        types.append(MambaBase)
    except ImportError:  # pragma: no cover - older vLLM without Mamba
        pass
    return tuple(types)


def _is_decoder_mixer(module: Any, mixer_types: tuple[type, ...]) -> bool:
    if not isinstance(module, mixer_types):
        return False
    # Encoder / cross-attention is not part of the decoder residual
    # stream.  Only ``Attention`` carries ``attn_type``; MLA and Mamba
    # mixers are decoder-only by construction.
    # ``attn_type`` is vLLM's ``AttentionType`` (a ``str`` Enum): compare by
    # value — ``str()`` of a str-Enum member is its *name* on Python ≥3.11.
    attn_type = getattr(module, "attn_type", None)
    return attn_type is None or attn_type == "decoder"


def _layers_from_registry(
    model: torch.nn.Module, registry: dict[str, Any]
) -> dict[int, torch.nn.Module]:
    """Map global decoder-layer index → module via ``static_forward_context``.

    Every attention-like layer registers there at construction under its
    full module prefix (e.g. ``model.layers.7.self_attn.attn``).  The
    decoder layer is the prefix truncated at its first integer segment,
    which is a *global* index even under pipeline parallelism (non-owned
    layers never construct, so they never register).
    """
    mixer_types = _decoder_mixer_types()
    layer_map: dict[int, torch.nn.Module] = {}
    for prefix, module in registry.items():
        if not _is_decoder_mixer(module, mixer_types):
            continue
        segments = prefix.split(".")
        idx_pos = next((i for i, seg in enumerate(segments) if seg.isdigit()), None)
        if idx_pos is None:
            raise LayerDiscoveryError(
                f"Registered layer {prefix!r} has no integer layer index in its "
                f"module path, so its decoder layer cannot be inferred."
            )
        layer_idx = int(segments[idx_pos])
        parent_path = ".".join(segments[: idx_pos + 1])
        try:
            layer = model.get_submodule(parent_path)
        except AttributeError as e:
            raise LayerDiscoveryError(
                f"Could not resolve decoder layer {parent_path!r} (from registered "
                f"layer {prefix!r}) on {type(model).__name__}."
            ) from e
        existing = layer_map.get(layer_idx)
        if existing is not None and existing is not layer:
            raise LayerDiscoveryError(
                f"Layer index {layer_idx} resolves to two different modules "
                f"({prefix!r} vs an earlier registry entry)."
            )
        layer_map[layer_idx] = layer
    return layer_map


def _layers_from_path_template(
    model: torch.nn.Module, template: str, total_layers: int
) -> dict[int, torch.nn.Module]:
    """Map layer index → module from a user-supplied path template.

    ``template`` is a dotted ``get_submodule`` path containing ``{i}``,
    e.g. ``"model.blocks.{i}"``.  Every index in ``range(total_layers)``
    must resolve; layers that resolve to ``PPMissingLayer`` belong to
    another pipeline stage and are skipped.
    """
    if "{i}" not in template:
        raise LayerDiscoveryError(
            f"{_LAYER_PATH_ENV}={template!r} must contain a '{{i}}' placeholder "
            f"for the layer index, e.g. 'model.layers.{{i}}'."
        )
    layer_map: dict[int, torch.nn.Module] = {}
    for i in range(total_layers):
        path = template.format(i=i)
        try:
            layer = model.get_submodule(path)
        except AttributeError as e:
            raise LayerDiscoveryError(
                f"{_LAYER_PATH_ENV}={template!r}: {path!r} does not exist on "
                f"{type(model).__name__}."
            ) from e
        if isinstance(layer, PPMissingLayer):
            continue
        layer_map[i] = layer
    return layer_map


def _discover_layer_modules(worker: Any) -> dict[int, torch.nn.Module]:
    """Map global decoder-layer index → decoder-layer module on this rank.

    Resolution order:

    1. ``VLLM_LENS_LAYER_PATH`` (a ``get_submodule`` path template such as
       ``"model.layers.{i}"``), if set — used exclusively.
    2. vLLM's ``static_forward_context`` registry (see
       :func:`_layers_from_registry`).

    The result is checked against the config's ``num_hidden_layers`` —
    every index must be in range and this rank must own exactly the
    layers pipeline parallelism assigns it.  Any failure raises
    :class:`LayerDiscoveryError` naming the fix; there is deliberately no
    silent fallback, since a wrong layer map yields wrong activations.
    """
    model = worker.model_runner.model
    total = _get_total_num_layers(worker)
    expected_local = int(worker.model_config.get_num_layers(worker.parallel_config))

    template = os.environ.get(_LAYER_PATH_ENV, "").strip()
    if template:
        layer_map = _layers_from_path_template(model, template, total)
        source = f"{_LAYER_PATH_ENV}={template!r}"
    else:
        registry = getattr(
            getattr(worker, "compilation_config", None), "static_forward_context", None
        )
        if not registry:
            raise LayerDiscoveryError(
                f"vLLM's static_forward_context registry is empty on "
                f"{type(model).__name__}; cannot discover decoder layers. "
                f"Set {_LAYER_PATH_ENV} (e.g. 'model.layers.{{i}}') to override."
            )
        layer_map = _layers_from_registry(model, registry)
        source = "static_forward_context registry"

    found = sorted(layer_map)
    out_of_range = [i for i in found if i < 0 or i >= total]
    if out_of_range or len(found) != expected_local:
        raise LayerDiscoveryError(
            f"Decoder-layer discovery via {source} found {len(found)} layer(s) "
            f"on this rank (indices {found}), but the model config declares "
            f"num_hidden_layers={total} with {expected_local} layer(s) on this "
            f"pipeline stage"
            + (f"; indices out of range: {out_of_range}" if out_of_range else "")
            + f". Set {_LAYER_PATH_ENV} to a get_submodule path template "
            f"(e.g. 'model.layers.{{i}}') to specify the decoder layers manually."
        )
    logger.info("vllm-lens: discovered %d decoder layer(s) via %s", len(found), source)
    return layer_map


def _get_total_num_layers(worker: Any) -> int:
    """Total decoder layers across all PP ranks (from the HF config)."""
    return int(worker.model_config.get_total_num_hidden_layers())


def _find_hook_configs(
    extension: HiddenStatesExtension,
    internal_req_id: str,
    extra_args: dict[str, Any] | None,
) -> list[Hook]:
    """Find all hook definitions that apply to an internal request ID.

    Checks three sources (in order):
    1. Per-request hooks keyed by external ID prefix (async path).
    2. Per-request hooks keyed by ``_hook_id`` sentinel (offline path).
    3. Persistent hooks (apply to every request).
    """
    results = _find_hook_configs_no_persistent(extension, internal_req_id, extra_args)
    results.extend(extension._persistent_hooks)
    return results


def _find_hook_configs_no_persistent(
    extension: HiddenStatesExtension,
    internal_req_id: str,
    extra_args: dict[str, Any] | None,
) -> list[Hook]:
    """Find per-request hook definitions only (excludes persistent hooks)."""
    results: list[Hook] = []
    for external_id, hooks in extension._hook_data.items():
        if internal_req_id.startswith(f"{external_id}-"):
            results.extend(hooks)
    if extra_args:
        hook_id = extra_args.get("_hook_id")
        if hook_id and hook_id in extension._hook_data:
            results.extend(extension._hook_data[hook_id])
    return results


class _SteeringLayer(NamedTuple):
    """Prepared additions and normalized directions for one request and layer. Each
    positional entry holds absolute positions, their values and whether they are matched."""

    add: torch.Tensor
    norm: torch.Tensor | None
    positional: list[tuple[list[int], torch.Tensor, bool]]


class _SteerSlice(NamedTuple):
    """One request's token slice and prepared steering."""

    start: int
    end: int
    abs_start: int
    data: _SteeringLayer


def _steer_pass(
    extension: HiddenStatesExtension,
) -> dict[int, list[_SteerSlice]] | None:
    """Group the token slices of the steered requests by layer, once per forward pass."""
    if not extension._steering_data:
        return None
    runner = extension.model_runner
    query_start_loc = _batch_layout(runner, warn=False)
    if query_start_loc is None:
        return None
    num_reqs = runner.input_batch.num_reqs
    bounds = query_start_loc[: num_reqs + 1].tolist()
    layers: dict[int, list[_SteerSlice]] = {}
    for i, req_id in enumerate(runner.input_batch.req_ids[:num_reqs]):
        start, end = bounds[i], bounds[i + 1]
        if start == end:
            continue
        request = runner.requests[req_id]
        extra = getattr(request.sampling_params, "extra_args", None)
        key = (extra or {}).get("_steering_id") or req_id.rpartition("-")[0]
        for layer, data in extension._steering_data.get(key, {}).items():
            layers.setdefault(layer, []).append(
                _SteerSlice(start, end, request.num_computed_tokens, data)
            )
    return layers or None


def _steering_buffers(
    spans: list[_SteerSlice],
    length: int,
    add_out: torch.Tensor | None = None,
    norm_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Combine request slices and zero padding in one write per buffer, then add the
    positional values."""
    zero = torch.zeros_like(spans[0].data.add)
    additions: list[torch.Tensor] = []
    directions: list[torch.Tensor] = []
    # Positional (token indices, values) for the add buffer (False) and the norm buffer (True).
    points: dict[bool, list[tuple[list[int], torch.Tensor]]] = {False: [], True: []}
    needs_norm = False
    cursor = 0
    for span in spans:
        if span.start > cursor:
            additions.append(zero.expand(span.start - cursor, -1))
            directions.append(zero.expand(span.start - cursor, -1))
        count = span.end - span.start
        add = span.data.add
        norm = span.data.norm if span.data.norm is not None else zero
        # A decode slice has one token, so the (1, hidden) rows need no expand.
        additions.append(add if count == 1 else add.expand(count, -1))
        directions.append(norm if count == 1 else norm.expand(count, -1))
        needs_norm |= span.data.norm is not None
        for positions, values, matched in span.data.positional:
            hits = [
                (span.start + pos - span.abs_start, i)
                for i, pos in enumerate(positions)
                if span.abs_start <= pos < span.abs_start + count
            ]
            if hits:
                target, source = zip(*hits)
                points[matched].append((list(target), values[list(source)]))
                needs_norm |= matched
        cursor = span.end
    if cursor < length:
        additions.append(zero.expand(length - cursor, -1))
        directions.append(zero.expand(length - cursor, -1))
    add = torch.cat(additions, out=add_out)
    norm = (
        torch.cat(directions, out=norm_out)
        if needs_norm or norm_out is not None
        else None
    )
    for target, values in points[False]:
        add.index_add_(0, torch.tensor(target, device=add.device), values)
    if norm is not None:
        for target, values in points[True]:
            norm.index_add_(0, torch.tensor(target, device=norm.device), values)
    return add, norm


def _steer(
    hidden: torch.Tensor,
    residual: torch.Tensor | None,
    add: torch.Tensor,
    norm: torch.Tensor | None,
) -> None:
    """Apply additions and directions scaled by the unsteered stream norm."""
    delta = add
    if norm is not None:
        stream = hidden if residual is None else hidden + residual
        stream_norm = torch.linalg.vector_norm(
            stream, dim=-1, keepdim=True, dtype=torch.float32
        )
        delta = delta + norm * stream_norm
    hidden += delta.to(hidden.dtype)


def _batch_layout(
    runner: Any, *, warn: bool
) -> Int[torch.Tensor, "num_reqs_plus1"] | None:  # type: ignore[reportUndefinedVariable]
    """``query_start_loc`` of this forward pass, or None when there is nothing to act on.
    With ``warn``, attention metadata without ``query_start_loc`` is logged."""
    if not is_forward_context_available() or runner.input_batch.num_reqs == 0:
        return None
    attn_metadata = get_forward_context().attn_metadata
    if isinstance(attn_metadata, list):
        attn_metadata = attn_metadata[0] if attn_metadata else None
    if attn_metadata is None:
        return None
    # Hybrid models (e.g. Qwen3-Next with GatedDeltaNet) have several metadata
    # entries, and some (like GDNAttentionMetadata) lack query_start_loc.
    for meta in attn_metadata.values():
        query_start_loc = getattr(meta, "query_start_loc", None)
        if query_start_loc is not None:
            return query_start_loc
    if warn:
        logger.warning(
            "No attention metadata with query_start_loc found "
            "(keys: %s). Skipping hook for this step.",
            list(attn_metadata.keys()),
        )
    return None


def _hook_inner(
    extension: HiddenStatesExtension,
    layer_idx: int,
    output: torch.Tensor | tuple[torch.Tensor, ...],
) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
    """Core hook logic, separated so _make_hook can wrap it in try/except."""
    runner = extension.model_runner
    query_start_loc = _batch_layout(runner, warn=True)
    if query_start_loc is None:
        return None
    num_reqs = runner.input_batch.num_reqs
    req_ids = runner.input_batch.req_ids

    # Apply the request slices planned for this layer.
    modified_output: torch.Tensor | tuple[torch.Tensor, ...] | None = None
    plan = extension._steer_pass
    if plan is not None and layer_idx in plan:
        target: torch.Tensor
        if isinstance(output, tuple):
            target, residual = output[0].clone(), output[1]
            modified_output = (target, *output[1:])
        else:
            target, residual = output.clone(), None
            modified_output = target
        add, norm = _steering_buffers(plan[layer_idx], target.shape[0])
        _steer(target, residual, add, norm)

    # --- Phase 2.5: run generic (post) hooks -------------------------
    # Per-request and persistent hooks are stored in separate context
    # dicts (per-request contexts are cleaned up after each request;
    # persistent ones accumulate).  Within each dict, contexts are keyed
    # by the hook's position in its category list — so a pre-hook and a
    # post-hook at different positions never share a HookContext, and the
    # returned result index ("0", "1", ...) is stable regardless of how
    # many pre/post hooks a request mixes.  Pre-hooks are handled in
    # _pre_hook_inner using the same position keys.
    per_req_hooks: list[list[Hook]] = []
    needs_hooks = False
    persistent_hooks = extension._persistent_hooks
    for i in range(num_reqs):
        req_id = req_ids[i]
        req_state = runner.requests.get(req_id)
        extra = (
            req_state.sampling_params.extra_args
            if req_state and req_state.sampling_params
            else None
        )
        hooks = _find_hook_configs_no_persistent(extension, req_id, extra)
        per_req_hooks.append(hooks)
        if hooks or persistent_hooks:
            needs_hooks = True

    if needs_hooks:
        # Compute hidden_states (summed if tuple) same as Phase 3 does.
        hook_src = modified_output if modified_output is not None else output
        if isinstance(hook_src, tuple):
            hook_hidden = (
                hook_src[0] + hook_src[1] if hook_src[1] is not None else hook_src[0]
            )
        else:
            hook_hidden = hook_src
        # Clone to avoid aliasing — hooks read/write this independently.
        hook_hidden = hook_hidden.clone()

        def _run_post_category(
            hooks: list[Hook],
            store: dict[str, dict[int, HookContext]],
            req_id: str,
            start: int,
            end: int,
        ) -> None:
            """Run the post-hooks in one category list at this layer.

            Contexts live in ``store[req_id][position]``, created lazily.
            """
            nonlocal modified_output
            for pos, hook in enumerate(hooks):
                if hook.pre or not hook.has_layer(layer_idx):
                    continue
                ctxs = store.setdefault(req_id, {})
                ctx = ctxs.get(pos)
                if ctx is None:
                    ctx = HookContext()
                    ctxs[pos] = ctx
                ctx.layer_idx = layer_idx
                ctx.seq_len = end - start
                ctx.model = runner.model
                ctx._prefetched = extension._prefetched_params

                result = hook.fn(ctx, hook_hidden[start:end])
                if result is not None:
                    modified_output = _replace_hook_output(
                        output, modified_output, hook_hidden, start, end, result
                    )

        for i in range(num_reqs):
            if not (persistent_hooks or per_req_hooks[i]):
                continue
            req_id = req_ids[i]
            start = int(query_start_loc[i].item())
            end = int(query_start_loc[i + 1].item())
            # Persistent hooks fire first (base layer); per-request hooks
            # see the persistent-modified state.
            _run_post_category(
                persistent_hooks,
                extension._persistent_hook_contexts,
                req_id,
                start,
                end,
            )
            _run_post_category(
                per_req_hooks[i], extension._hook_contexts, req_id, start, end
            )

    # --- Phase 3: capture activations (rank 0 only) -----------------
    if getattr(extension, "_should_capture", True):
        capture_src = modified_output if modified_output is not None else output
        hidden_states: Float[torch.Tensor, "total_tokens hidden_dim"]  # type: ignore[reportUndefinedVariable]
        if isinstance(capture_src, tuple):
            if capture_src[1] is not None:
                hidden_states = capture_src[0] + capture_src[1]
            else:
                hidden_states = capture_src[0]
        else:
            hidden_states = capture_src

        for i in range(num_reqs):
            req_id = req_ids[i]
            req_state = runner.requests.get(req_id)
            if req_state is None or req_state.sampling_params is None:
                continue
            extra = req_state.sampling_params.extra_args
            if not extra:
                continue

            output_residual_stream = extra.get("output_residual_stream")
            if output_residual_stream is None:
                continue
            if (
                isinstance(output_residual_stream, list)
                and layer_idx not in output_residual_stream
            ):
                continue

            start = query_start_loc[i].item()
            end = query_start_loc[i + 1].item()
            activation: Float[torch.Tensor, "seq_len hidden_dim"] = hidden_states[  # type: ignore[reportUndefinedVariable]
                start:end
            ].cpu()

            if req_id not in extension._captured_states:
                extension._captured_states[req_id] = {}
            layer_states = extension._captured_states[req_id]
            if layer_idx not in layer_states:
                layer_states[layer_idx] = []
            layer_states[layer_idx].append(activation)

    return modified_output


def _pre_hook_inner(
    extension: HiddenStatesExtension,
    layer_idx: int,
    input_tensor: torch.Tensor,
) -> torch.Tensor | None:
    """Run pre-hooks (hook.pre=True) on the layer input.

    Only runs generic hooks — steering and activation capture are
    post-hook operations and are not affected.
    """
    runner = extension.model_runner
    query_start_loc = _batch_layout(runner, warn=False)
    if query_start_loc is None:
        return None
    num_reqs = runner.input_batch.num_reqs
    req_ids = runner.input_batch.req_ids

    # Pre-hooks share the same context stores as post-hooks, keyed by the
    # hook's position in its category list.  A hook at a given position is
    # either pre or post (never both), so pre and post never collide on the
    # same key — this is what lets a request mix pre- and post-hooks safely.
    persistent_hooks = extension._persistent_hooks
    modified = False
    working = input_tensor

    def _run_pre_category(
        hooks: list[Hook],
        store: dict[str, dict[int, HookContext]],
        req_id: str,
        start: int,
        end: int,
    ) -> None:
        """Run the pre-hooks in one category list at this layer."""
        nonlocal working, modified
        for pos, hook in enumerate(hooks):
            if not hook.pre or not hook.has_layer(layer_idx):
                continue
            ctxs = store.setdefault(req_id, {})
            hctx = ctxs.get(pos)
            if hctx is None:
                hctx = HookContext()
                ctxs[pos] = hctx
            hctx.layer_idx = layer_idx
            hctx.seq_len = end - start
            hctx.model = runner.model
            hctx._prefetched = extension._prefetched_params

            result = hook.fn(hctx, working[start:end])
            if result is not None:
                if not modified:
                    working = input_tensor.clone()
                    modified = True
                working[start:end] = result

    for i in range(num_reqs):
        req_id = req_ids[i]
        req_state = runner.requests.get(req_id)
        extra = (
            req_state.sampling_params.extra_args
            if req_state and req_state.sampling_params
            else None
        )
        per_req = _find_hook_configs_no_persistent(extension, req_id, extra)
        if not any(h.pre for h in persistent_hooks) and not any(h.pre for h in per_req):
            continue

        start = int(query_start_loc[i].item())
        end = int(query_start_loc[i + 1].item())
        _run_pre_category(
            persistent_hooks, extension._persistent_hook_contexts, req_id, start, end
        )
        _run_pre_category(per_req, extension._hook_contexts, req_id, start, end)

    return working if modified else None


def _make_hook(extension: HiddenStatesExtension, layer_idx: int) -> Callable:
    """Create a forward hook closure for a specific layer index."""

    def hook(
        _module: torch.nn.Module,
        _input: object,
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
        """Forward hook: apply steering vectors then capture activations.

        Returns the modified output if any steering was applied, ``None``
        otherwise (so PyTorch leaves the original output untouched).
        """
        try:
            return _hook_inner(extension, layer_idx, output)
        except Exception:
            logger.warning(
                "vllm-lens hook error on layer %d, skipping", layer_idx, exc_info=True
            )
            return None

    return hook


def _make_pre_hook(extension: HiddenStatesExtension, layer_idx: int) -> Callable:
    """Create a forward pre-hook closure for a specific layer index.

    vLLM decoder layers have signature
    ``forward(positions, hidden_states, residual)`` — the hidden states
    are at ``args[1]``, not ``args[0]``.
    """

    def hook(
        _module: torch.nn.Module,
        args: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...] | None:
        """Forward pre-hook: run user pre-hooks on the layer input."""
        try:
            # hidden_states is args[1] (args[0] is positions).
            hidden = args[1]
            result = _pre_hook_inner(extension, layer_idx, hidden)
            if result is not None:
                return args[:1] + (result,) + args[2:]
            return None
        except Exception:
            logger.warning(
                "vllm-lens pre-hook error on layer %d, skipping",
                layer_idx,
                exc_info=True,
            )
            return None

    return hook


def _warn_qk_skipped(extension: HiddenStatesExtension, reason: str) -> None:
    """Log once per worker when Q/K capture has to bail out of a forward.

    The hook otherwise returns silently and the client only sees a missing
    ``qk_layers`` key, which points at the wrong cause.
    """
    if extension._qk_skip_warned:
        return
    extension._qk_skip_warned = True
    logger.warning(
        "vllm-lens: output_qk capture skipped a forward pass (%s); attention "
        "Q/K for this request may be incomplete or missing.",
        reason,
    )


def _qk_hook_inner(
    extension: HiddenStatesExtension,
    layer_idx: int,
    attn_module: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    """Capture post-RoPE Q/K at the input of a vLLM ``Attention`` module.

    Unlike residual-stream capture, this runs on **every** TP rank:
    ``Attention`` inputs are sharded by head (each rank sees only its own
    slice of query/kv heads), so all ranks must contribute and the plugin
    concatenates along the head dimension at merge time.
    """
    if not is_forward_context_available():
        return

    runner = extension.model_runner
    num_reqs = runner.input_batch.num_reqs
    if num_reqs == 0:
        return

    query = kwargs.get("query", args[0] if len(args) > 0 else None)
    key = kwargs.get("key", args[1] if len(args) > 1 else None)
    if query is None or key is None:
        return

    ctx = get_forward_context()
    attn_metadata = ctx.attn_metadata
    if isinstance(attn_metadata, list):
        attn_metadata = attn_metadata[0] if attn_metadata else None
    if attn_metadata is None:
        _warn_qk_skipped(extension, "forward context carries no attn_metadata")
        return
    # attn_metadata is keyed by layer name — prefer this layer's own entry,
    # falling back to the first entry with query_start_loc (hybrid models).
    query_start_loc: torch.Tensor | None = None
    if isinstance(attn_metadata, dict):
        meta = attn_metadata.get(getattr(attn_module, "layer_name", ""))
        if meta is not None and hasattr(meta, "query_start_loc"):
            query_start_loc = getattr(meta, "query_start_loc")
        else:
            for _meta in attn_metadata.values():
                if hasattr(_meta, "query_start_loc"):
                    query_start_loc = getattr(_meta, "query_start_loc")
                    break
    else:
        query_start_loc = getattr(attn_metadata, "query_start_loc", None)
    if query_start_loc is None:
        _warn_qk_skipped(extension, "attn_metadata has no query_start_loc")
        return

    req_ids = runner.input_batch.req_ids

    # Inputs are (num_tokens, heads * head_size), post-RoPE (the model
    # applies rotary embeddings before calling the Attention wrapper);
    # 3-D inputs are legal per the wrapper's contract, so reshape both.
    num_heads = attn_module.num_heads
    num_kv_heads = attn_module.num_kv_heads
    head_size = attn_module.head_size
    q3 = query.reshape(-1, num_heads, head_size)
    k3 = key.reshape(-1, num_kv_heads, head_size)

    for i in range(num_reqs):
        req_id = req_ids[i]
        req_state = runner.requests.get(req_id)
        if req_state is None or req_state.sampling_params is None:
            continue
        extra = req_state.sampling_params.extra_args
        if not extra:
            continue

        output_qk = extra.get("output_qk")
        if output_qk is None:
            continue
        # vllm_xargs passes values as strings; parse JSON lists.
        if isinstance(output_qk, str):
            try:
                output_qk = json.loads(output_qk)
            except (json.JSONDecodeError, ValueError):
                pass  # treat as truthy (capture all layers)
        if isinstance(output_qk, list) and layer_idx not in output_qk:
            continue

        start = int(query_start_loc[i].item())
        end = int(query_start_loc[i + 1].item())

        if req_id not in extension._captured_qk:
            extension._captured_qk[req_id] = {}
        layer_states = extension._captured_qk[req_id]
        if layer_idx not in layer_states:
            layer_states[layer_idx] = {"q": [], "k": []}
        layer_states[layer_idx]["q"].append(q3[start:end].cpu())
        layer_states[layer_idx]["k"].append(k3[start:end].cpu())


def _make_qk_hook(
    extension: HiddenStatesExtension, layer_idx: int, attn_module: Any
) -> Callable:
    """Create a forward pre-hook closure for an ``Attention`` module."""

    def hook(
        _module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        """Forward pre-hook: capture this layer's Q/K for opted-in requests."""
        try:
            _qk_hook_inner(extension, layer_idx, attn_module, args, kwargs)
        except Exception:
            logger.warning(
                "vllm-lens qk hook error on layer %d, skipping",
                layer_idx,
                exc_info=True,
            )
        return None

    return hook


def _normalize_sliding_window(value: Any) -> tuple[int, int]:
    """Normalize a sliding-window spec to vLLM's ``(left, right)`` form."""
    if value is None:
        return (-1, -1)
    if isinstance(value, int):
        return (-1, -1) if value < 0 else (value - 1, 0)
    left, right = value
    return (int(left), int(right))


def _to_float_list(value: Any) -> list[float] | None:
    """Convert an optional per-head tensor/sequence to a JSON-safe list."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return [float(x) for x in value.detach().float().cpu().tolist()]
    return [float(x) for x in value]


class HiddenStatesExtension:
    """Mixin injected into vLLM's GPU Worker at runtime.

    Configured via the ``worker_extension_cls`` engine arg. vLLM dynamically
    adds this class as a base of Worker
    (``Worker.__bases__ += (HiddenStatesExtension,)``), so ``self`` is the
    Worker instance and its methods are callable via
    ``collective_rpc("method_name")``.

    It doesn't extend Worker directly — vLLM handles that injection.
    """

    if TYPE_CHECKING:
        model_runner: Any  # Provided by Worker at runtime
        model_config: Any
        compilation_config: Any
        rank: int
        parallel_config: ParallelConfig

    # Per-request captured activations:
    # internal_req_id → { layer_idx → [tensor, ...] }
    _captured_states: dict[
        str,
        dict[int, list[Float[torch.Tensor, "seq_len hidden_dim"]]],  # type: ignore[reportUndefinedVariable]
    ] = {}
    _hooks_installed: bool = False

    # Prepared steering by request key and layer.
    _steering_data: dict[str, dict[int, _SteeringLayer]] = {}

    # Per-request hook definitions:
    # key (external_req_id or _hook_id) → list of Hook
    _hook_data: dict[str, list[Hook]] = {}

    # The steering slices of the current forward pass.
    _steer_pass: dict[int, list[_SteerSlice]] | None = None

    # Persistent hooks (apply to every request, not auto-cleaned):
    _persistent_hooks: list[Hook] = []

    # Per-request hook contexts, keyed by internal request ID then by the
    # hook's position in the per-request hook list:
    # internal_req_id → { hook_position → HookContext }
    _hook_contexts: dict[str, dict[int, HookContext]] = {}

    # Persistent hook contexts (separate from per-request to avoid cleanup
    # conflicts), keyed the same way by position in the persistent hook list:
    # internal_req_id → { hook_position → HookContext }
    _persistent_hook_contexts: dict[str, dict[int, HookContext]] = {}

    # Whether this rank should capture activations (only TP rank 0).
    _should_capture: bool = True

    # Per-request captured attention Q/K (all TP ranks — heads are sharded):
    # internal_req_id → { layer_idx → {"q": [tensor, ...], "k": [tensor, ...]} }
    _captured_qk: dict[str, dict[int, dict[str, list[torch.Tensor]]]] = {}
    _qk_hooks_installed: bool = False
    _qk_skip_warned: bool = False

    # Per-layer kernel metadata recorded at install time, keyed by global
    # layer index (scale, sliding window, soft-cap, ... — see install_qk_hooks).
    _qk_layer_meta: dict[int, dict[str, Any]] = {}

    def install_hooks(self) -> None:
        """Register a forward hook on every decoder layer. Idempotent.

        Hooks are installed on **all** TP ranks because steering must
        modify hidden states everywhere.  Activation *capture* is gated
        to rank 0 only via ``_should_capture``.

        Requires ``enforce_eager=True`` in engine args — otherwise
        ``@support_torch_compile`` would compile the forward graph and
        hooks won't fire.
        """
        if self._hooks_installed:
            return
        self._hooks_installed = True
        # Reset to instance-level dicts (class-level defaults are shared).
        # Do NOT reset _persistent_hooks — they may have been set via
        # set_persistent_hooks() before the first generate call.
        self._captured_states = {}
        self._steering_data = {}
        self._hook_data = {}
        self._steer_pass = None
        if not isinstance(self.__dict__.get("_persistent_hooks"), list):
            self._persistent_hooks = []
        self._hook_contexts = {}
        self._persistent_hook_contexts = {}

        # Only rank 0 captures — residual streams are replicated across
        # TP ranks after all-reduce, so the data is identical.
        tp_size = self.parallel_config.tensor_parallel_size
        self._should_capture = tp_size <= 1 or self.rank % tp_size == 0

        # Hooks must be installed on ALL ranks so steering vectors are
        # applied everywhere (not just rank 0).
        layer_map = _discover_layer_modules(self)
        for layer_idx, layer in sorted(layer_map.items()):
            layer.register_forward_pre_hook(_make_pre_hook(self, layer_idx))
            layer.register_forward_hook(_make_hook(self, layer_idx))

        runner = self.model_runner
        model_forward = runner._model_forward

        def _model_forward(*args: Any, **kwargs: Any) -> Any:
            """Plan this forward pass's steering, then run it."""
            try:
                self._steer_pass = _steer_pass(self)
            except Exception:
                # As for a bad hook: one request must not stop the engine.
                logger.warning("Steering failed for this step", exc_info=True)
                self._steer_pass = None
            try:
                return model_forward(*args, **kwargs)
            finally:
                # A forward pass outside this wrapper (a dummy run) steers nothing.
                self._steer_pass = None

        runner._model_forward = _model_forward

    # ------------------------------------------------------------------
    # Steering data management (called via collective_rpc)
    # ------------------------------------------------------------------

    def set_steering_data(self, key: str, pickled_data: bytes) -> None:
        """Validate and prepare steering vectors for one request."""
        sv_list: list[SteeringVector] = pickle.loads(pickled_data)

        device = next(self.model_runner.model.parameters()).device
        dtype = next(self.model_runner.model.parameters()).dtype

        num_layers = _get_total_num_layers(self)
        hidden = self.model_runner.model_config.get_hidden_size()
        prepared: dict[int, _SteeringLayer] = {}

        for sv in sv_list:
            for idx in sv.layer_indices:
                if idx < 0 or idx >= num_layers:
                    raise ValueError(
                        f"layer_index {idx} out of range [0, {num_layers})"
                    )
            if sv.activations.shape[-1] != hidden:
                raise ValueError(
                    f"vector width {sv.activations.shape[-1]}, model width {hidden}"
                )

            values = sv.activations.to(device=device, dtype=dtype).float()
            if sv.norm_match:
                values = values / (values.norm(dim=-1, keepdim=True) + 1e-6)
            values = values * sv.scale
            for layer, index in sv.layer_index_map.items():
                if layer not in prepared:
                    prepared[layer] = _SteeringLayer(
                        torch.zeros((1, hidden), dtype=torch.float32, device=device),
                        None,
                        [],
                    )
                data = prepared[layer]
                if values.dim() == 3:
                    count = values.shape[1]
                    positions = sv.position_indices
                    if positions is None:
                        positions = list(range(count))
                    data.positional.append(
                        (list(positions[:count]), values[index], sv.norm_match)
                    )
                elif sv.norm_match:
                    norm = (
                        data.norm
                        if data.norm is not None
                        else torch.zeros_like(data.add)
                    )
                    norm.add_(values[index])
                    prepared[layer] = data._replace(norm=norm)
                else:
                    data.add.add_(values[index])
        self._steering_data[key] = prepared

    def clear_steering_data(self, key: str) -> None:
        """Remove steering data for a completed request."""
        self._steering_data.pop(key, None)

    def clear_captured_states(self, external_req_id: str) -> None:
        """Remove captured activations without returning them.

        Called in the ``finally`` block of ``_patched_generate`` to clean
        up leaked state when a request is aborted or the client disconnects
        before ``get_captured_states`` is called.  On normal completion this
        is a no-op because ``get_captured_states`` already ``.pop()``-ed
        the entry.
        """
        prefix = f"{external_req_id}-"
        for req_id in list(self._captured_states):
            if req_id.startswith(prefix):
                del self._captured_states[req_id]
                logger.debug("Cleared leaked activations for %s", req_id)

    def _build_payload(
        self, internal_req_id: str
    ) -> dict[str, dict[str, "Float[torch.Tensor, 'n_layers total_pos hidden_dim']"]]:  # type: ignore[reportUndefinedVariable]
        """Materialise the stacked-tensor payload for one internal request id.

        Pops the entry from ``_captured_states`` so successive calls do not
        re-emit the same data. Shared by :meth:`get_captured_states` and
        :meth:`get_captured_states_batch`.
        """
        layer_dict = self._captured_states.pop(internal_req_id)
        sorted_indices = sorted(layer_dict.keys())
        per_layer: list[Float[torch.Tensor, "total_pos hidden_dim"]] = [  # type: ignore[reportUndefinedVariable]
            torch.cat(layer_dict[idx], dim=0) for idx in sorted_indices
        ]
        stacked: Float[torch.Tensor, "n_layers total_pos hidden_dim"] = (  # type: ignore[reportUndefinedVariable]
            torch.stack(per_layer, dim=0)
        )
        if stacked.is_cuda:
            stacked = stacked.cpu()
        return {"activations": {"residual_stream": stacked}}

    def get_captured_states(self, external_req_id: str) -> bytes | None:
        """Retrieve captured activations for a specific request.

        Matches by ``"{external_req_id}-"`` prefix because vLLM internally
        transforms the user-provided ``request_id`` into
        ``"{request_id}-{random_suffix}"``. So ``"req-0"`` matches
        ``"req-0-a1b2c3d4"`` but NOT ``"req-00-b5c6d7e8"``.

        Moves tensors to CPU and serializes via pickle + zstd for safe ZMQ
        transport (the compression matters most when the response crosses
        the network in the OpenAI/Inspect HTTP path).

        Returns a dict when deserialized::

            {
                "activations": {
                    "residual_stream": Tensor,  # (n_layers, total_pos, d_model)
                }
            }

        Layers are stacked in ascending order along dim 0.
        Removes the request's data after retrieval.
        """
        prefix = f"{external_req_id}-"
        for req_id in list(self._captured_states):
            if req_id.startswith(prefix):
                payload = self._build_payload(req_id)
                return _ZSTD_COMPRESSOR.compress(pickle.dumps(payload))
        return None

    def get_captured_states_batch(self, external_req_ids: list[str]) -> bytes | None:
        """Retrieve captured activations for many requests in one RPC.

        Equivalent to calling :meth:`get_captured_states` once per id, but
        emits a single payload covering every request that has data. At
        large batch sizes the per-request ``collective_rpc`` roundtrip is
        the dominant cost on the offline ``LLM.generate`` path; batching it
        cuts N round-trips to one.

        Returns ``pickle.dumps({external_req_id: payload, ...})`` where
        each ``payload`` is ``{"activations": {"residual_stream": Tensor}}``.
        Missing or unmatched ids are simply absent; ``None`` if nothing
        matched at all.

        We deliberately don't ``zstd``-compress here: this RPC only fires
        on the offline ``LLM.generate`` path, which is in-process IPC,
        not HTTP. ``get_captured_states`` keeps zstd for the OpenAI/Inspect
        path where the response crosses the network.
        """
        if not external_req_ids:
            return None
        out: dict[str, dict[str, Any]] = {}
        # Walk live state once and bucket by external id; matches the
        # ``"{external_req_id}-"`` prefix rule from get_captured_states.
        for req_id in list(self._captured_states):
            for external_req_id in external_req_ids:
                if req_id.startswith(f"{external_req_id}-"):
                    out[external_req_id] = self._build_payload(req_id)
                    break
        if not out:
            return None
        return pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL)

    def _debug_captured_states_count(self) -> int:
        """Return the number of entries in _captured_states (for testing)."""
        return len(self._captured_states)

    def _debug_layer_discovery(self) -> list[int]:
        """Global indices of decoder layers found by discovery (testing)."""
        return sorted(_discover_layer_modules(self))

    # ------------------------------------------------------------------
    # Attention Q/K capture (called via collective_rpc)
    # ------------------------------------------------------------------

    def _select_qk_attention_modules(self) -> dict[int, Any]:
        """Global layer index → the decoder ``Attention`` module inside it.

        Reuses decoder-layer discovery (registry or ``VLLM_LENS_LAYER_PATH``,
        count-checked) so Q/K layer indices are exactly the residual-stream
        ones.  Layers without a softmax-attention wrapper (Mamba /
        linear-attention blocks in hybrids) are omitted; MLA is refused.
        """
        from vllm.model_executor.layers.attention import Attention, MLAAttention

        layer_map = _discover_layer_modules(self)
        selected: dict[int, Any] = {}
        for layer_idx, layer in layer_map.items():
            attn_modules = [
                m
                for m in layer.modules()
                if isinstance(m, (Attention, MLAAttention))
                and _is_decoder_mixer(m, (Attention, MLAAttention))
            ]
            if any(isinstance(m, MLAAttention) for m in attn_modules):
                raise RuntimeError(
                    "output_qk is not supported for MLA models (DeepSeek-style "
                    "multi-head latent attention): the Attention wrapper never "
                    "sees materialized per-head Q/K, so there is nothing to "
                    "capture. Residual-stream capture and steering still work."
                )
            if len(attn_modules) > 1:
                raise RuntimeError(
                    f"Decoder layer {layer_idx} contains {len(attn_modules)} "
                    "decoder Attention modules; output_qk expects exactly one."
                )
            if attn_modules:
                selected[layer_idx] = attn_modules[0]
        return selected

    # -- raw Q/K reference capture (testing only) ------------------------------
    #
    # Plain torch pre-hooks on the same ``Attention`` modules, recording the
    # *whole* batch's Q/K per forward with no per-request slicing, merging or
    # serialization.  Tests compare the production capture against this
    # bit-for-bit (``torch.equal``) to verify the plumbing exactly — hook
    # firing under chunked prefill, query_start_loc slicing, TP head-shard
    # order, KV-replication dedupe, PP layer concat, wire round-trip.

    def _debug_raw_qk_install(self, layer_indices: list[int]) -> list[int]:
        self._debug_raw_qk = {i: {"q": [], "k": []} for i in layer_indices}
        self._debug_raw_qk_handles = []
        selected = self._select_qk_attention_modules()
        for layer_idx in layer_indices:
            if layer_idx not in selected:  # not on this PP stage
                continue
            module = selected[layer_idx]

            def hook(_m, args, kwargs, *, _idx=layer_idx, _mod=module):
                query = kwargs.get("query", args[0] if args else None)
                key = kwargs.get("key", args[1] if len(args) > 1 else None)
                if query is None or key is None:
                    return None
                store = self._debug_raw_qk[_idx]
                store["q"].append(
                    query.reshape(-1, _mod.num_heads, _mod.head_size).cpu()
                )
                store["k"].append(
                    key.reshape(-1, _mod.num_kv_heads, _mod.head_size).cpu()
                )
                return None

            self._debug_raw_qk_handles.append(
                module.register_forward_pre_hook(hook, with_kwargs=True)
            )
        return sorted(selected)

    def _debug_raw_qk_pop(self) -> bytes:
        """Return and clear the raw capture: pickled ``{layer: {q, k}}`` plus ranks."""
        from vllm.distributed import parallel_state as ps

        out: dict[str, Any] = {
            "tp_rank": int(ps.get_tensor_model_parallel_rank()),
            "pp_rank": int(ps.get_pp_group().rank_in_group),
            "layers": {},
        }
        for idx, st in self._debug_raw_qk.items():
            if st["q"]:
                out["layers"][idx] = {
                    "q": torch.cat(st["q"], dim=0),
                    "k": torch.cat(st["k"], dim=0),
                }
            st["q"].clear()
            st["k"].clear()
        return pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL)

    def _debug_raw_qk_remove(self) -> None:
        for h in getattr(self, "_debug_raw_qk_handles", []):
            h.remove()
        self._debug_raw_qk_handles = []
        self._debug_raw_qk = {}

    def install_qk_hooks(self) -> None:
        """Register a pre-hook on every decoder ``Attention`` module. Idempotent.

        Separate from :meth:`install_hooks` so residual-stream/steering
        users pay no extra per-forward cost.  Installed — and capturing —
        on **every** TP rank, because ``Attention`` inputs are sharded by
        head (unlike the residual stream, which is replicated post
        all-reduce).

        Also records, per layer, the exact kernel parameters needed to
        replay the attention softmax offline (``vllm_lens.attention``):
        vLLM has already normalized model-specific conventions (custom
        scales, sliding-window forms, ...) into these values.
        """
        if self._qk_hooks_installed:
            return

        selected = self._select_qk_attention_modules()

        if not selected:
            raise RuntimeError(
                "output_qk found no decoder attention layers to hook "
                "(is this an attention-free model?)."
            )

        # Reset to instance-level state (class-level defaults are shared).
        self._captured_qk = {}
        self._qk_layer_meta = {}
        self._qk_hooks_installed = True

        model_config = getattr(self, "model_config", None)
        try:
            total_kv_heads = int(model_config.get_total_num_kv_heads())  # type: ignore[union-attr]
        except Exception:
            total_kv_heads = 0

        from vllm.distributed import parallel_state as ps

        tp_size = self.parallel_config.tensor_parallel_size
        self._qk_tp_rank = int(ps.get_tensor_model_parallel_rank())
        self._qk_pp_rank = int(ps.get_pp_group().rank_in_group)
        self._qk_tp_size = int(tp_size)

        for layer_idx, module in sorted(selected.items()):
            impl = module.impl
            self._qk_layer_meta[layer_idx] = {
                "layer_name": getattr(module, "layer_name", ""),
                "scale": float(getattr(impl, "scale", None) or module.head_size**-0.5),
                "logits_soft_cap": float(getattr(impl, "logits_soft_cap", None) or 0.0),
                "sliding_window": list(
                    _normalize_sliding_window(getattr(impl, "sliding_window", None))
                ),
                "alibi_slopes": _to_float_list(getattr(impl, "alibi_slopes", None)),
                "sinks": _to_float_list(getattr(impl, "sinks", None)),
                "use_alibi_sqrt": bool(getattr(module, "use_alibi_sqrt", False)),
                "num_heads_local": int(module.num_heads),
                "num_kv_heads_local": int(module.num_kv_heads),
                "head_size": int(module.head_size),
                "num_kv_heads_total": total_kv_heads or int(module.num_kv_heads),
            }
            module.register_forward_pre_hook(
                _make_qk_hook(self, layer_idx, module), with_kwargs=True
            )

    def _build_qk_payload(self, internal_req_id: str) -> dict[str, Any]:
        """Materialise the stacked Q/K payload for one internal request id.

        Pops the entry so successive calls do not re-emit the same data.
        Q stacks to ``(n_layers, total_pos, local_q_heads, head_size)`` and
        K to ``(n_layers, total_pos, local_kv_heads, head_size)``.
        """
        layer_dict = self._captured_qk.pop(internal_req_id)
        sorted_indices = sorted(layer_dict.keys())
        q_layers = [torch.cat(layer_dict[i]["q"], dim=0) for i in sorted_indices]
        k_layers = [torch.cat(layer_dict[i]["k"], dim=0) for i in sorted_indices]
        if len({t.shape for t in q_layers}) > 1 or len({t.shape for t in k_layers}) > 1:
            # Heterogeneous head counts across captured layers (exotic
            # hybrids). Stacking would silently misalign — refuse loudly.
            raise RuntimeError(
                "output_qk captured layers with heterogeneous Q/K shapes; "
                "capture a single layer at a time for this model."
            )
        return {
            "qk": {
                "q": torch.stack(q_layers, dim=0),
                "k": torch.stack(k_layers, dim=0),
                "layers": sorted_indices,
                "meta": [self._qk_layer_meta[i] for i in sorted_indices],
                "tp_rank": getattr(self, "_qk_tp_rank", 0),
                "pp_rank": getattr(self, "_qk_pp_rank", 0),
                "tp_size": getattr(self, "_qk_tp_size", 1),
            }
        }

    def get_captured_qk(self, external_req_id: str) -> bytes | None:
        """Retrieve captured Q/K for a request (zstd + pickle).

        Same ``"{external_req_id}-"`` prefix-match convention as
        :meth:`get_captured_states`.  Every TP rank returns its own head
        shard, tagged with its ranks; the plugin merges along the head
        dimension.  Removes the request's data after retrieval.
        """
        prefix = f"{external_req_id}-"
        for req_id in list(self._captured_qk):
            if req_id.startswith(prefix):
                payload = self._build_qk_payload(req_id)
                return _ZSTD_COMPRESSOR.compress(pickle.dumps(payload))
        return None

    def get_captured_qk_batch(self, external_req_ids: list[str]) -> bytes | None:
        """Batched :meth:`get_captured_qk` for the offline path (plain pickle)."""
        if not external_req_ids:
            return None
        out: dict[str, dict[str, Any]] = {}
        for req_id in list(self._captured_qk):
            for external_req_id in external_req_ids:
                if req_id.startswith(f"{external_req_id}-"):
                    out[external_req_id] = self._build_qk_payload(req_id)
                    break
        if not out:
            return None
        return pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL)

    def clear_captured_qk(self, external_req_id: str) -> None:
        """Remove captured Q/K without returning it (abort/disconnect cleanup)."""
        prefix = f"{external_req_id}-"
        for req_id in list(self._captured_qk):
            if req_id.startswith(prefix):
                del self._captured_qk[req_id]
                logger.debug("Cleared leaked Q/K capture for %s", req_id)

    def _debug_captured_qk_count(self) -> int:
        """Return the number of entries in _captured_qk (for testing)."""
        return len(self._captured_qk)

    # ------------------------------------------------------------------
    # Hook data management (called via collective_rpc)
    # ------------------------------------------------------------------

    def set_hook_data(self, key: str, pickled_data: bytes) -> None:
        """Receive and store hook definitions for a request.

        Called via ``collective_rpc`` before generation begins.  Unpickles
        the list of ``Hook`` instances (using cloudpickle for the callable
        ``fn``), validates layer indices against the model, and stores them
        keyed by *key* (an external request ID or ``_hook_id`` sentinel).
        """
        hooks: list[Hook] = cloudpickle.loads(pickled_data)
        num_layers = _get_total_num_layers(self)
        for hook in hooks:
            for idx in hook.layer_indices:
                if idx < 0 or idx >= num_layers:
                    raise ValueError(
                        f"layer_index {idx} out of range [0, {num_layers})"
                    )
        self._hook_data[key] = hooks

    def get_hook_results(self, external_req_id: str) -> bytes | None:
        """Retrieve hook results (``ctx.saved`` dicts) for a request.

        Returns from ALL ranks (including PP ranks that own different
        layers).  The plugin merges results across ranks.
        Matches by ``"{external_req_id}-"`` prefix on ``_hook_contexts``.
        Returns ``{str(hook_position): ctx.saved}`` pickled, where
        ``hook_position`` indexes the per-request hook list.
        """
        prefix = f"{external_req_id}-"
        for req_id in list(self._hook_contexts):
            if req_id.startswith(prefix):
                contexts = self._hook_contexts.pop(req_id)
                saved_dicts = {str(pos): ctx.saved for pos, ctx in contexts.items()}
                return pickle.dumps(saved_dicts)
        return None

    def clear_hook_data(self, key: str) -> None:
        """Remove hook definitions for a completed request."""
        self._hook_data.pop(key, None)

    def clear_hook_contexts(self, external_req_id: str) -> None:
        """Remove hook contexts for a completed or aborted request.

        Prefix-match cleanup, same pattern as ``clear_captured_states``.
        """
        prefix = f"{external_req_id}-"
        for req_id in list(self._hook_contexts):
            if req_id.startswith(prefix):
                del self._hook_contexts[req_id]

    # ------------------------------------------------------------------
    # Persistent hook management (called via collective_rpc)
    # ------------------------------------------------------------------

    def set_persistent_hooks(self, pickled_data: bytes) -> None:
        """Append hooks that apply to every subsequent request.

        Accepts cloudpickle'd ``list[Hook]``.  Validates layer indices.
        Appends to existing persistent hooks (call ``clear_persistent_hooks``
        first for a clean slate).  Also ensures forward hooks are installed
        on the model layers.
        """
        self.install_hooks()
        hooks: list[Hook] = cloudpickle.loads(pickled_data)
        num_layers = _get_total_num_layers(self)
        for hook in hooks:
            for idx in hook.layer_indices:
                if idx < 0 or idx >= num_layers:
                    raise ValueError(
                        f"layer_index {idx} out of range [0, {num_layers})"
                    )
        self._persistent_hooks.extend(hooks)

    def get_all_hook_results(self) -> bytes | None:
        """Retrieve accumulated persistent hook contexts from all requests.

        Returns from ALL ranks (for PP support).  Does NOT clear — call
        ``clear_persistent_hooks`` explicitly.

        Returns pickled ``{internal_req_id: {hook_idx_str: ctx.saved}}``.
        """
        if not self._persistent_hook_contexts:
            return None
        results: dict[str, dict[str, dict[str, Any]]] = {}
        for req_id, contexts in self._persistent_hook_contexts.items():
            results[req_id] = {str(pos): ctx.saved for pos, ctx in contexts.items()}
        return pickle.dumps(results)

    def clear_persistent_hooks(self) -> None:
        """Remove persistent hooks and all accumulated contexts."""
        self._persistent_hooks = []
        self._persistent_hook_contexts = {}

    def clear_persistent_hook_results(self) -> None:
        """Drop accumulated persistent-hook contexts, keeping hooks registered.

        ``get_all_hook_results`` never drains, so results pile up across
        requests. This clears them without unregistering the hooks (so a
        fitted lens does not need re-uploading), letting a client bound
        accumulation by clearing between turns.
        """
        self._persistent_hook_contexts = {}

    # ------------------------------------------------------------------
    # Parameter prefetch (called via collective_rpc — all ranks in sync)
    # ------------------------------------------------------------------

    _prefetched_params: dict[str, torch.Tensor] = {}

    def prefetch_parameters(self, names: list[str]) -> None:
        """Pre-fetch and gather parameters across TP and PP ranks.

        Safe to call PP collectives here because ``collective_rpc``
        runs on all ranks simultaneously.  Results are stored in
        ``_prefetched_params`` for use by ``HookContext.get_parameter``.
        """
        import torch.distributed as dist

        from vllm.distributed.parallel_state import get_pp_group, get_tp_group
        from vllm.model_executor.models.utils import PPMissingLayer

        model = self.model_runner.model
        tp_group = get_tp_group()
        pp_group = get_pp_group()

        for name in names:
            # Traverse to find the parameter.
            obj: Any = model
            parts = name.split(".")
            is_local = True
            for attr in parts:
                obj = getattr(obj, attr)
                if isinstance(obj, PPMissingLayer):
                    is_local = False
                    break

            param: torch.Tensor | None = None
            if is_local:
                local_t = torch.as_tensor(obj)

                # TP gather if sharded; otherwise reuse the existing tensor.
                module: Any = model
                for attr in parts[:-1]:
                    module = getattr(module, attr)
                tp_size = getattr(module, "tp_size", 1)
                if tp_size > 1:
                    gathered = [torch.empty_like(local_t) for _ in range(tp_size)]
                    dist.all_gather(gathered, local_t, group=tp_group.device_group)
                    gather_dim = getattr(module, "gather_dim", 0)
                    param = torch.cat(gathered, dim=gather_dim)
                else:
                    param = local_t  # no copy — reference to existing parameter

            # PP broadcast — safe here because all ranks are in this RPC.
            if pp_group.world_size > 1:
                has_it = torch.tensor(
                    [1 if is_local else 0], device="cuda", dtype=torch.int32
                )
                all_has = [torch.zeros_like(has_it) for _ in range(pp_group.world_size)]
                dist.all_gather(all_has, has_it, group=pp_group.device_group)
                source_pp = next(i for i, t in enumerate(all_has) if t.item() == 1)
                source_global = pp_group.ranks[source_pp]

                if param is None:
                    # Receive shape + dtype.
                    meta = torch.zeros(3, device="cuda", dtype=torch.int64)
                    dist.broadcast(meta, src=source_global, group=pp_group.device_group)
                    ndim = int(meta[0].item())
                    dtype = _DTYPE_LIST[int(meta[1].item())]
                    shape_t = torch.zeros(ndim, device="cuda", dtype=torch.int64)
                    dist.broadcast(
                        shape_t, src=source_global, group=pp_group.device_group
                    )
                    shape = tuple(int(s) for s in shape_t.tolist())
                    param = torch.empty(shape, device="cuda", dtype=dtype)
                else:
                    meta = torch.tensor(
                        [param.ndim, _dtype_to_idx(param.dtype), 0],
                        device="cuda",
                        dtype=torch.int64,
                    )
                    dist.broadcast(meta, src=source_global, group=pp_group.device_group)
                    shape_t = torch.tensor(
                        list(param.shape),
                        device="cuda",
                        dtype=torch.int64,
                    )
                    dist.broadcast(
                        shape_t, src=source_global, group=pp_group.device_group
                    )

                dist.broadcast(param, src=source_global, group=pp_group.device_group)

            assert param is not None, f"Parameter {name!r} not found on any rank"
            self._prefetched_params[name] = param

    def clear_prefetched_params(self) -> None:
        """Remove all pre-fetched parameters."""
        self._prefetched_params = {}

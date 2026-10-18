"""
Worker extension that captures residual-stream activations from
configurable layers during transformer forward passes, and optionally
applies steering vectors (activation additions) to modify the residual
stream in-flight.

Uses PyTorch forward hooks on each decoder layer for concurrency-safe,
per-request activation capture and steering.  Each hook checks the
request's ``extra_args["output_residual_stream"]`` to decide whether to
capture, and decodes the request's steering vectors and hooks from its ``extra_args``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple

import cloudpickle
import numpy as np
import torch
import zstandard as zstd
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.models.utils import PPMissingLayer

from vllm_lens._helpers._hook_output import _replace_hook_output
from vllm_lens._helpers._positions import (
    POSITIONS_KEY,
    POSITIONS_META_KEY,
    order_positions,
    positions_in_slice,
)
from vllm_lens._helpers._serialize import decode_layers, deserialize_directions
from vllm_lens._helpers.types import (
    HOOKS_PAYLOAD,
    STEERING_PAYLOAD,
    Hook,
    HookContext,
    SteeringVector,
)

if TYPE_CHECKING:
    from jaxtyping import Float, Int
    from vllm.config import ParallelConfig

# A child of vLLM's logger, so vLLM's handler, format and VLLM_LOGGING_LEVEL apply.
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


def _decode_vectors(
    extension: HiddenStatesExtension, raw: bytes
) -> list[SteeringVector]:
    """Unpickle a request's vectors onto the model device and dtype."""
    param = next(extension.model_runner.model.parameters())
    return [
        sv.model_copy(
            update={
                "activations": sv.activations.to(device=param.device, dtype=param.dtype)
            }
        )
        for sv in pickle.loads(raw)
    ]


def _request_payload(
    extension: HiddenStatesExtension,
    req_state: Any,
    req_id: str,
    key: str,
) -> list:
    """The decoded ``extra_args[key]`` payload of a request, or [] without one. Requests with
    the same bytes share one decoded copy, held until the last of them leaves the runner."""
    held = extension._request_payloads.get((key, req_id))
    if held is None:
        params = req_state.sampling_params if req_state is not None else None
        raw = (params.extra_args or {}).get(key) if params is not None else None
        if raw is None:
            return []
        held = hashlib.blake2b(raw, digest_size=16).digest()
        entry = extension._payloads.get(held)
        if entry is None:
            try:
                if key == STEERING_PAYLOAD:
                    decoded = _decode_vectors(extension, raw)
                else:
                    decoded = cloudpickle.loads(raw)
            except Exception:
                # As for a bad hook: one request must not stop the engine.
                logger.warning("Could not decode %s of %s", key, req_id, exc_info=True)
                decoded = []
            entry = extension._payloads[held] = [decoded, 0]
        entry[1] += 1
        extension._request_payloads[(key, req_id)] = held
    return extension._payloads[held][0]


def _release_payloads(extension: HiddenStatesExtension, req_id: str) -> None:
    """Drop a finished request's hold on its decoded payloads."""
    for key in (STEERING_PAYLOAD, HOOKS_PAYLOAD):
        held = extension._request_payloads.pop((key, req_id), None)
        if held is None:
            continue
        entry = extension._payloads[held]
        entry[1] -= 1
        if entry[1] == 0:
            del extension._payloads[held]
            if extension._steer_table is not None:
                extension._steer_table.release(held)


def norm_match(
    residual: torch.Tensor,
    steering: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Scale a steering vector to match the L2 norm of the residual stream.

    Norm matching approach from the Activation Oracles paper
    (arXiv:2512.15674):

        h'_i = h_i + ‖h_i‖ · v_i / ‖v_i‖

    This rescales the steering vector so its magnitude matches the
    residual before addition, ensuring activations of varying provenance
    are automatically scaled to a consistent magnitude.
    """
    r_norm = residual.float().norm(dim=-1, keepdim=True)
    v_norm = steering.float().norm(dim=-1, keepdim=True)
    return (steering * (r_norm / (v_norm + eps))).to(residual.dtype)


# Rows after the request rows that hold the position-specific vectors of one step.
SCRATCH_ROWS = 512


@dataclass
class _SteerTable:
    """float32 steering rows per layer, ``scale`` folded in: ``add`` is added as is, ``norm`` is
    a unit row scaled by the stream norm. Row 0 is zero, rows below ``n_rows`` hold one payload
    each, and the ``SCRATCH_ROWS`` rows after them are written again each step."""

    device: torch.device
    hidden: int
    n_rows: int
    add: dict[int, torch.Tensor] = field(default_factory=dict)
    norm: dict[int, torch.Tensor] = field(default_factory=dict)
    free: list[int] = field(default_factory=list)
    # The row of each payload digest: 0 when it has no 2-D vector or cannot be steered.
    rows: dict[bytes, int] = field(default_factory=dict)
    # The position-specific (3-D) vectors of each payload digest.
    positional: dict[bytes, list[SteeringVector]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.free = list(range(self.n_rows - 1, 0, -1))

    def layer(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The ``add`` and ``norm`` tables of ``layer``, allocated on first use."""
        if layer not in self.add:
            shape = (self.n_rows + SCRATCH_ROWS, self.hidden)
            self.add[layer] = torch.zeros(
                shape, dtype=torch.float32, device=self.device
            )
            self.norm[layer] = torch.zeros(
                shape, dtype=torch.float32, device=self.device
            )
        return self.add[layer], self.norm[layer]

    def row(self, key: bytes, configs: list[SteeringVector]) -> int:
        """The row of ``key``, written from its 2-D vectors on first use. A payload that cannot
        be written is logged and steers nothing, as one bad request must not stop the engine."""
        row = self.rows.get(key)
        if row is not None:
            return row
        flat = [cfg for cfg in configs if cfg.activations.dim() == 2]
        row = 0
        try:
            for cfg in configs:
                if cfg.activations.shape[-1] != self.hidden:
                    raise ValueError(
                        f"vector width {cfg.activations.shape[-1]}, model width {self.hidden}"
                    )
                for layer in cfg.layer_index_map:
                    self.layer(layer)
            if flat and not self.free:
                logger.warning(
                    "All %d steering rows are in use; a request is not steered",
                    self.n_rows - 1,
                )
                self.rows[key] = 0
                return 0
            if flat:
                row = self.rows[key] = self.free.pop()
                for table in (*self.add.values(), *self.norm.values()):
                    table[row].zero_()
            for cfg in flat:
                for layer, index in cfg.layer_index_map.items():
                    vec = cfg.activations[index].float()
                    if cfg.norm_match:
                        vec = vec / (vec.norm() + 1e-6)
                    tables = self.norm if cfg.norm_match else self.add
                    tables[layer][row] += vec * cfg.scale
        except Exception:
            logger.warning(
                "Steering failed for %s; it is not steered", key, exc_info=True
            )
            self.release(key)
            self.rows[key] = 0
            return 0
        self.rows[key] = row
        positional = [cfg for cfg in configs if cfg.activations.dim() != 2]
        if positional:
            self.positional[key] = positional
        return row

    def release(self, key: bytes) -> None:
        """Give the row of ``key`` back."""
        row = self.rows.pop(key, None)
        self.positional.pop(key, None)
        if row:
            self.free.append(row)


class _SteerPass(NamedTuple):
    """The steering of one forward pass: the table row of each token, and the layers to steer."""

    rows: torch.Tensor
    layers: frozenset[int]


def _steer_pass(extension: HiddenStatesExtension) -> _SteerPass | None:
    """The token -> row index of this forward pass: a steered request's tokens take its row,
    and each in-step position of a position-specific vector takes a scratch row."""
    runner = extension.model_runner
    query_start_loc = _batch_layout(runner, warn=False)
    if query_start_loc is None:
        return None
    num_reqs = runner.input_batch.num_reqs
    req_ids = runner.input_batch.req_ids
    steered: list[tuple[int, int, list[SteeringVector], list[SteeringVector]]] = []
    for i in range(num_reqs):
        configs = _request_payload(
            extension, runner.requests.get(req_ids[i]), req_ids[i], STEERING_PAYLOAD
        )
        if not configs:
            continue
        if extension._steer_table is None:
            extension._steer_table = _SteerTable(
                device=next(runner.model.parameters()).device,
                hidden=runner.model_config.get_hidden_size(),
                n_rows=1 + runner.scheduler_config.max_num_seqs,
            )
        table = extension._steer_table
        digest = extension._request_payloads[(STEERING_PAYLOAD, req_ids[i])]
        row = table.row(digest, configs)
        positional = table.positional.get(digest, [])
        if row or positional:
            steered.append((i, row, configs, positional))
    if not steered:
        return None
    table = extension._steer_table
    assert table is not None
    bounds = _host_layout(runner, query_start_loc, num_reqs)
    index = np.zeros(bounds[num_reqs], dtype=np.int32)
    layers: set[int] = set()
    # The base row of each scratch row, and each vector's (scratch row, position) uses.
    bases: list[int] = []
    uses: dict[int, tuple[SteeringVector, list[int], list[int]]] = {}
    skipped = 0
    for i, row, configs, positional in steered:
        start, end = bounds[i], bounds[i + 1]
        index[start:end] = row
        layers.update(layer for cfg in configs for layer in cfg.layer_index_map)
        if not positional:
            continue
        abs_start = runner.requests[req_ids[i]].num_computed_tokens
        at: dict[int, list[tuple[SteeringVector, int]]] = {}
        for cfg in positional:
            n_pos = cfg.activations.shape[1]
            positions = (
                range(n_pos) if cfg.position_indices is None else cfg.position_indices
            )
            for pi, pos in enumerate(positions[:n_pos]):
                if abs_start <= pos < abs_start + end - start:
                    at.setdefault(pos, []).append((cfg, pi))
        for pos in sorted(at):
            if len(bases) == SCRATCH_ROWS:
                skipped += 1
                continue
            index[start + pos - abs_start] = table.n_rows + len(bases)
            for cfg, pi in at[pos]:
                use = uses.setdefault(id(cfg), (cfg, [], []))
                use[1].append(len(bases))
                use[2].append(pi)
            bases.append(row)
    if skipped:
        logger.warning(
            "%d steered positions in one step are over the %d scratch rows; "
            "they are not steered",
            skipped,
            SCRATCH_ROWS,
        )
    if bases:
        device = table.device
        base = torch.tensor(bases, device=device)
        targets = torch.arange(table.n_rows, table.n_rows + len(bases), device=device)
        for layer, add in table.add.items():
            rows = {False: add[base], True: table.norm[layer][base]}
            for cfg, slots, pis in uses.values():
                if layer not in cfg.layer_index_map:
                    continue
                vecs = cfg.activations[cfg.layer_index_map[layer]][pis].float()
                if cfg.norm_match:
                    vecs = vecs / (vecs.norm(dim=-1, keepdim=True) + 1e-6)
                rows[cfg.norm_match].index_add_(
                    0, torch.tensor(slots, device=device), vecs * cfg.scale
                )
            add.index_copy_(0, targets, rows[False])
            table.norm[layer].index_copy_(0, targets, rows[True])
    host = torch.from_numpy(index)
    if table.device.type == "cuda":
        host = host.pin_memory()
    return _SteerPass(host.to(table.device, non_blocking=True), frozenset(layers))


def _steer(
    hidden: torch.Tensor,
    residual: torch.Tensor | None,
    add: torch.Tensor,
    norm: torch.Tensor,
    index: torch.Tensor,
) -> None:
    """Add each token's row in place: ``add`` as is, ``norm`` times the norm of the unsteered
    stream (``hidden + residual``), in float32."""
    rows = index[: hidden.shape[0]]
    n = rows.shape[0]
    stream = hidden[:n] if residual is None else hidden[:n] + residual[:n]
    stream_norm = torch.linalg.vector_norm(
        stream, dim=-1, keepdim=True, dtype=torch.float32
    )
    hidden[:n] += (add[rows] + norm[rows] * stream_norm).to(hidden.dtype)


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


def _host_layout(
    runner: Any, query_start_loc: torch.Tensor, num_reqs: int
) -> list[int]:
    """``query_start_loc`` on the host: the runner's pinned copy when the forward pass uses the
    runner's buffer, which needs no GPU sync; else a read of the device tensor."""
    buffer = getattr(runner, "query_start_loc", None)
    gpu = getattr(buffer, "gpu", None)
    if gpu is not None and gpu.data_ptr() == query_start_loc.data_ptr():
        return buffer.np[: num_reqs + 1].tolist()  # type: ignore[union-attr]
    return query_start_loc[: num_reqs + 1].tolist()


def _token_layout(
    spans: list[tuple[int, int]], labels: list[int], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """The batch index of every token in ``spans`` and the label of its span, on ``device``,
    through one pinned copy that does not wait for the stream."""
    lengths = np.asarray([end - start for start, end in spans], dtype=np.int64)
    starts = np.repeat(
        np.asarray([start for start, _ in spans], dtype=np.int64), lengths
    )
    offsets = np.arange(lengths.sum(), dtype=np.int64) - np.repeat(
        np.cumsum(lengths) - lengths, lengths
    )
    layout = torch.from_numpy(
        np.stack(
            [starts + offsets, np.repeat(np.asarray(labels, dtype=np.int64), lengths)]
        )
    )
    if device.type == "cuda":
        layout = layout.pin_memory()
    tokens, segments = layout.to(device, non_blocking=True)
    return tokens, segments


def _to_host(rows: torch.Tensor) -> torch.Tensor:
    """Queue a copy of ``rows`` into pinned host memory and return it without waiting.
    The engine syncs each step's output before the request finishes."""
    if not rows.is_cuda:
        return rows
    host = torch.empty(rows.shape, dtype=rows.dtype, pin_memory=True)
    host.copy_(rows, non_blocking=True)
    return host


def _capture_rows(
    extension: HiddenStatesExtension,
    layer_idx: int,
    hidden_states: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> None:
    """Store the requested rows of one layer: every or the selected positions, the sum of the
    prompt rows for a mean, or the projections onto a registered direction set."""
    runner = extension.model_runner
    num_reqs = runner.input_batch.num_reqs
    req_ids = runner.input_batch.req_ids
    bounds: list[int] | None = None
    # Full and position-selected captures: each request's batch rows.
    picked: list[tuple[str, np.ndarray]] = []
    pooled: list[tuple[str, int, int, int]] = []
    projected: dict[str, list[tuple[str, int, int]]] = {}
    for i in range(num_reqs):
        req_id = req_ids[i]
        req_state = runner.requests.get(req_id)
        if req_state is None or req_state.sampling_params is None:
            continue
        extra = req_state.sampling_params.extra_args
        if not extra or extra.get("output_residual_stream") is None:
            continue
        layers = decode_layers(extra["output_residual_stream"])
        if layers is not None and layer_idx not in layers:
            continue
        if bounds is None:
            bounds = _host_layout(runner, query_start_loc, num_reqs)
        start, end = bounds[i], bounds[i + 1]
        # Positions [first, ...) of the request are rows [start, ...) of this step.
        first = req_state.num_computed_tokens
        selector = extra.get(POSITIONS_KEY)
        projection = extra.get("output_residual_stream_project")
        # The front end validates these values.
        if extra.get("output_residual_stream_pool") == "mean":
            high = min(first + end - start, req_state.num_prompt_tokens)
            if high > first:
                pooled.append((req_id, start, start - first + high, first))
                # The mean is the sum over the prompt divided at collection.
                extension._pooled[req_id] = (
                    hidden_states.dtype,
                    req_state.num_prompt_tokens,
                )
        elif projection is not None:
            projected.setdefault(projection, []).append((req_id, start, end))
        elif selector is not None:
            positions = positions_in_slice(
                selector, first, end - start, req_state.num_prompt_tokens
            )
            extension._captured_positions.setdefault(req_id, {}).setdefault(
                layer_idx, []
            ).extend(positions)
            picked.append(
                (req_id, np.asarray(positions, dtype=np.int64) + start - first)
            )
        else:
            picked.append((req_id, np.arange(start, end, dtype=np.int64)))

    for projection, spans in projected.items():
        directions = extension._direction_tables.get(projection)
        if directions is None:
            # The set was cleared while the requests ran.
            logger.warning(
                "No projection %r for requests %s; their capture is skipped",
                projection,
                [req_id for req_id, _, _ in spans],
            )
            continue
        rows = torch.cat([hidden_states[start:end] for _, start, end in spans]).float()
        # Each position is stored as its N projections and its norm, in float32.
        host = _to_host(
            torch.cat([rows @ directions.T, rows.norm(dim=-1, keepdim=True)], dim=1)
        )
        for (req_id, start, end), piece in zip(
            spans, host.split([end - start for _, start, end in spans])
        ):
            extension._projections.add(req_id)
            extension._captured_states.setdefault(req_id, {}).setdefault(
                layer_idx, []
            ).append(piece)
    if pooled:
        tokens, segments = _token_layout(
            [(low, high) for _, low, high, _ in pooled],
            list(range(len(pooled))),
            hidden_states.device,
        )
        # The sums stay on the device until collection, so the hook does not wait for the GPU.
        sums = torch.zeros(
            (len(pooled), hidden_states.shape[-1]),
            dtype=torch.float32,
            device=hidden_states.device,
        ).index_add_(0, segments, hidden_states.index_select(0, tokens).float())
        for row, (req_id, _, _, first) in enumerate(pooled):
            total = sums[row : row + 1]
            states = extension._captured_states.setdefault(req_id, {})
            # A prefill from position 0 (also after a preemption) starts a new sum.
            if first > 0 and layer_idx in states:
                total = states[layer_idx][0] + total
            states[layer_idx] = [total]
    if picked:
        index = torch.from_numpy(np.concatenate([rows for _, rows in picked]))
        if hidden_states.is_cuda:
            index = index.pin_memory()
        host = _to_host(
            hidden_states.index_select(
                0, index.to(hidden_states.device, non_blocking=True)
            )
        )
        for (req_id, _), piece in zip(
            picked, host.split([len(rows) for _, rows in picked])
        ):
            extension._captured_states.setdefault(req_id, {}).setdefault(
                layer_idx, []
            ).append(piece)


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
    bounds = _host_layout(runner, query_start_loc, num_reqs)

    # --- Phase 1: steering, from the rows this forward pass planned --
    modified_output: torch.Tensor | tuple[torch.Tensor, ...] | None = None
    plan = extension._steer_pass
    table = extension._steer_table
    if plan is not None and table is not None and layer_idx in plan.layers:
        add, norm = table.layer(layer_idx)
        if isinstance(output, tuple):
            modified_output = (output[0].clone(), *output[1:])
            _steer(modified_output[0], output[1], add, norm, plan.rows)
        else:
            modified_output = output.clone()
            _steer(modified_output, None, add, norm, plan.rows)

    # --- Phase 2: run generic (post) hooks ---------------------------
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
        hooks = _request_payload(
            extension, runner.requests.get(req_id), req_id, HOOKS_PAYLOAD
        )
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
            start, end = bounds[i], bounds[i + 1]
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

        _capture_rows(extension, layer_idx, hidden_states, query_start_loc)

    return modified_output


def _pre_hook_inner(
    extension: HiddenStatesExtension,
    layer_idx: int,
    input_tensor: torch.Tensor,
) -> torch.Tensor | None:
    """Run generic pre-hooks on the layer input."""
    runner = extension.model_runner
    query_start_loc = _batch_layout(runner, warn=False)
    if query_start_loc is None:
        return None
    num_reqs = runner.input_batch.num_reqs
    req_ids = runner.input_batch.req_ids
    bounds: list[int] | None = None

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
        per_req = _request_payload(
            extension, runner.requests.get(req_id), req_id, HOOKS_PAYLOAD
        )
        if not any(h.pre for h in persistent_hooks) and not any(h.pre for h in per_req):
            continue

        if bounds is None:
            bounds = _host_layout(runner, query_start_loc, num_reqs)
        start, end = bounds[i], bounds[i + 1]
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
    bounds: list[int] | None = None

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

        if extra.get("output_qk") is None:
            continue
        layers = decode_layers(extra["output_qk"])
        if layers is not None and layer_idx not in layers:
            continue

        if bounds is None:
            bounds = _host_layout(runner, query_start_loc, num_reqs)
        start, end = bounds[i], bounds[i + 1]

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
    # Pooled capture: _captured_states holds one float32 sum row per layer.
    # internal_req_id -> (capture dtype, divisor of the sum)
    _pooled: dict[str, tuple[torch.dtype, int]] = {}
    # Position-selected capture: the absolute position of each captured row.
    # internal_req_id -> { layer_idx -> [position, ...] }
    _captured_positions: dict[str, dict[int, list[int]]] = {}
    # Projected capture: the internal ids whose rows hold projections and a norm.
    _projections: set[str] = set()
    # Registered direction sets by name.
    _direction_tables: dict[str, torch.Tensor] = {}
    _hooks_installed: bool = False

    # Decoded steering and hook payloads by the hash of their bytes, with the number of
    # requests that hold each: digest -> [decoded, count].
    _payloads: dict[bytes, list[Any]] = {}
    # The digest each running request holds: (payload key, internal_req_id) -> digest.
    _request_payloads: dict[tuple[str, str], bytes] = {}
    # The steering rows of the stored payloads, and the steering of the current forward pass.
    _steer_table: _SteerTable | None = None
    _steer_pass: _SteerPass | None = None

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

    def _reset_state(self) -> None:
        """Create the per-instance state and decide which rank captures."""
        # Instance-level dicts: the class-level defaults are shared. _persistent_hooks is
        # kept, as set_persistent_hooks() may run before the first generate call.
        self._captured_states = {}
        self._pooled = {}
        self._captured_positions = {}
        self._projections = set()
        self._direction_tables = {}
        self._payloads = {}
        self._request_payloads = {}
        self._steer_table = None
        self._steer_pass = None
        if not isinstance(self.__dict__.get("_persistent_hooks"), list):
            self._persistent_hooks = []
        self._hook_contexts = {}
        self._persistent_hook_contexts = {}

        # Only rank 0 captures — residual streams are replicated across
        # TP ranks after all-reduce, so the data is identical.
        tp_size = self.parallel_config.tensor_parallel_size
        self._should_capture = tp_size <= 1 or self.rank % tp_size == 0

        # The runner calls this for every request that leaves it, finished or aborted.
        runner = self.model_runner
        removed = runner._on_request_state_removed

        def _on_request_state_removed(req_id: str, req_state: Any) -> None:
            _release_payloads(self, req_id)
            removed(req_id, req_state)

        runner._on_request_state_removed = _on_request_state_removed
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

    def install_hooks(self) -> None:
        """Install decoder-layer hooks on every rank and reset per-instance state once."""
        if self._hooks_installed:
            return
        self._hooks_installed = True
        self._reset_state()

        # Hooks must be installed on ALL ranks so steering vectors are
        # applied everywhere (not just rank 0).
        layer_map = _discover_layer_modules(self)
        for layer_idx, layer in sorted(layer_map.items()):
            layer.register_forward_pre_hook(_make_pre_hook(self, layer_idx))
            layer.register_forward_hook(_make_hook(self, layer_idx))

    def clear_request_state(self, external_ids: list[str] | None) -> None:
        """Drop the captures and hook contexts of ``external_ids``; None drops those of every
        request (a failed offline call)."""
        wanted = None if external_ids is None else set(external_ids)
        for store in (
            self._captured_states,
            self._pooled,
            self._captured_positions,
            self._captured_qk,
            self._hook_contexts,
        ):
            for req_id in list(store):
                if wanted is None or req_id.rpartition("-")[0] in wanted:
                    del store[req_id]
        self._projections = {
            req_id
            for req_id in self._projections
            if wanted is not None and req_id.rpartition("-")[0] not in wanted
        }

    # ------------------------------------------------------------------
    # Projection directions (called via collective_rpc)
    # ------------------------------------------------------------------

    def set_projection(self, name: str, payload: str) -> None:
        """Hold the ``(N, hidden)`` directions of a ``serialize_tensor`` payload under
        ``name``; requests name it in ``output_residual_stream_project``."""
        # Installing the hooks resets the worker state, so it must come first.
        self.install_hooks()
        device = next(self.model_runner.model.parameters()).device
        self._direction_tables[name] = deserialize_directions(payload).to(device)

    def clear_projection(self, name: str | None) -> None:
        """Drop the directions under ``name``; None drops every set."""
        if name is None:
            self._direction_tables = {}
        else:
            self._direction_tables.pop(name, None)

    def _build_payload(self, internal_req_id: str) -> dict[str, dict[str, Any]]:
        """Pop one internal request id's capture as host tensors: a pooled capture is one mean
        row per layer in the capture dtype, a projected capture its scores and its norms, and a
        position-selected capture one row per position in ascending order."""
        layer_dict = self._captured_states.pop(internal_req_id)
        pooled = self._pooled.pop(internal_req_id, None)
        positions = self._captured_positions.pop(internal_req_id, None)
        projected = internal_req_id in self._projections
        self._projections.discard(internal_req_id)
        layers = sorted(layer_dict)
        if pooled is not None:
            # Each layer holds one float32 sum row; divide to get the mean.
            dtype, divisor = pooled
            sums = torch.stack([layer_dict[layer][0] for layer in layers])
            return {
                "activations": {"residual_stream": (sums / divisor).to(dtype).cpu()}
            }
        stacked = torch.stack([torch.cat(layer_dict[layer]) for layer in layers])
        activations: dict[str, Any] = {}
        if positions is not None:
            # A position computed twice (after a preemption) keeps its latest row.
            ordered, keep = order_positions(positions[layers[0]])
            if keep != list(range(stacked.shape[1])):
                stacked = stacked[:, keep]
            activations[POSITIONS_META_KEY] = ordered
        if projected:
            # Each row holds the N projections, then the norm.
            activations["residual_stream_projection"] = stacked[..., :-1]
            activations["residual_stream_norm"] = stacked[..., -1]
        else:
            activations["residual_stream"] = stacked
        return {"activations": activations}

    def get_captured_states(self, external_req_id: str) -> bytes | None:
        """Retrieve captured activations for a specific request.

        vLLM turns the user's ``request_id`` into ``"{request_id}-{random_suffix}"``, so
        this matches the internal id before its last ``-``.

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
        for req_id in list(self._captured_states):
            if req_id.rpartition("-")[0] == external_req_id:
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
        wanted = set(external_req_ids)
        for req_id in list(self._captured_states):
            external_req_id = req_id.rpartition("-")[0]
            if external_req_id in wanted and external_req_id not in out:
                out[external_req_id] = self._build_payload(req_id)
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

        Matches the external id as :meth:`get_captured_states` does.  Every TP rank returns its own head
        shard, tagged with its ranks; the plugin merges along the head
        dimension.  Removes the request's data after retrieval.
        """
        for req_id in list(self._captured_qk):
            if req_id.rpartition("-")[0] == external_req_id:
                payload = self._build_qk_payload(req_id)
                return _ZSTD_COMPRESSOR.compress(pickle.dumps(payload))
        return None

    def get_captured_qk_batch(self, external_req_ids: list[str]) -> bytes | None:
        """Batched :meth:`get_captured_qk` for the offline path (plain pickle)."""
        if not external_req_ids:
            return None
        out: dict[str, dict[str, Any]] = {}
        wanted = set(external_req_ids)
        for req_id in list(self._captured_qk):
            external_req_id = req_id.rpartition("-")[0]
            if external_req_id in wanted and external_req_id not in out:
                out[external_req_id] = self._build_qk_payload(req_id)
        if not out:
            return None
        return pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL)

    def _debug_captured_qk_count(self) -> int:
        """Return the number of entries in _captured_qk (for testing)."""
        return len(self._captured_qk)

    def get_hook_results(self, external_req_id: str) -> bytes | None:
        """Retrieve hook results (``ctx.saved`` dicts) for a request.

        Returns from ALL ranks (including PP ranks that own different
        layers).  The plugin merges results across ranks.
        Matches the external id: the internal id before its last ``-``.
        Returns ``{str(hook_position): ctx.saved}`` pickled, where
        ``hook_position`` indexes the per-request hook list.
        """
        for req_id in list(self._hook_contexts):
            if req_id.rpartition("-")[0] == external_req_id:
                contexts = self._hook_contexts.pop(req_id)
                saved_dicts = {str(pos): ctx.saved for pos, ctx in contexts.items()}
                return pickle.dumps(saved_dicts)
        return None

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

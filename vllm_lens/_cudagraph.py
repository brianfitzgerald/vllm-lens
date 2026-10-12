"""CUDA-graph mode: serve vllm-lens requests without forcing ``enforce_eager``.
The README section "CUDA graphs" describes each path that replaces a forward hook."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from vllm_lens._helpers.types import Hook, SteeringVector

logger = logging.getLogger(__name__)

# The key of these settings in ``VllmConfig.additional_config``.
CONFIG_KEY = "vllm_lens"

# In the stored settings, so in vLLM's compile cache key; nothing reads it. Increase
# it when a change alters what a compiled graph contains and no setting changes.
GRAPH_VERSION = 1


@dataclass(frozen=True)
class LensGraphConfig:
    """The CUDA-graph settings of one engine, read from the environment once.
    ``VllmConfig.additional_config`` holds them for the front end and the workers."""

    enabled: bool = False
    capture_layers: tuple[int, ...] = ()
    steer_layers: tuple[int, ...] = ()

    @classmethod
    def from_env(cls) -> LensGraphConfig:
        """Read ``VLLM_LENS_CUDAGRAPH`` and the ``VLLM_LENS_*_LAYERS`` lists."""
        flag = os.environ.get("VLLM_LENS_CUDAGRAPH", "").strip().lower()
        layers: dict[str, tuple[int, ...]] = {}
        for name in ("VLLM_LENS_CAPTURE_LAYERS", "VLLM_LENS_STEER_LAYERS"):
            parts = [
                part for part in os.environ.get(name, "").split(",") if part.strip()
            ]
            try:
                layers[name] = tuple(sorted({int(part) for part in parts}))
            except ValueError:
                raise ValueError(
                    f"{name} must be a comma-separated list of layer numbers, "
                    f"got {os.environ[name]!r}."
                ) from None
        return cls(
            enabled=flag in ("1", "true", "yes", "on") or any(layers.values()),
            capture_layers=layers["VLLM_LENS_CAPTURE_LAYERS"],
            steer_layers=layers["VLLM_LENS_STEER_LAYERS"],
        )

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> LensGraphConfig:
        """Read the settings that ``to_additional_config`` stored."""
        stored = (getattr(vllm_config, "additional_config", None) or {}).get(CONFIG_KEY)
        if not isinstance(stored, dict):
            return cls()
        # A user can write this key too, so a missing field takes its default.
        return cls(
            enabled=bool(stored.get("enabled", False)),
            capture_layers=tuple(sorted(stored.get("capture_layers", ()))),
            steer_layers=tuple(sorted(stored.get("steer_layers", ()))),
        )

    def to_additional_config(
        self, existing: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """``existing`` with these settings, or without them when the mode is off."""
        if not self.enabled and CONFIG_KEY not in (existing or {}):
            return existing
        others = {
            key: value for key, value in (existing or {}).items() if key != CONFIG_KEY
        }
        if not self.enabled:
            return others
        return {**others, CONFIG_KEY: {**asdict(self), "graph_version": GRAPH_VERSION}}

    def reject_unserved(
        self,
        residual_stream: Any,
        steering_layers: set[int],
        hooks: list[Hook],
        qk: bool = False,
    ) -> None:
        """Raise ``ValueError`` for a request that this CUDA-graph server cannot serve.
        The arguments are the request's capture value, steering layers, hooks, Q/K flag."""
        if not self.enabled:
            return
        if hooks or qk:
            raise ValueError(
                "VLLM_LENS_CUDAGRAPH is set, so the vllm-lens forward hooks do not "
                "run: hooks and Q/K capture are not available. Unset "
                "VLLM_LENS_CUDAGRAPH to serve this request in eager mode."
            )
        unsteered = sorted(steering_layers - set(self.steer_layers))
        if unsteered:
            raise ValueError(
                f"apply_steering_vectors names layer(s) {unsteered}, but under "
                "VLLM_LENS_CUDAGRAPH this server steers only the layers in "
                f"VLLM_LENS_STEER_LAYERS: {list(self.steer_layers)}."
            )
        if residual_stream is None:
            return
        if isinstance(residual_stream, str):
            try:
                residual_stream = json.loads(residual_stream)
            except (json.JSONDecodeError, ValueError):
                pass
        # "All layers" would return only the named ones, with no layer labels.
        if not isinstance(residual_stream, list):
            raise ValueError(
                "Under VLLM_LENS_CUDAGRAPH, output_residual_stream must be a list of "
                f"layers from VLLM_LENS_CAPTURE_LAYERS: {list(self.capture_layers)}."
            )
        outside = sorted(set(residual_stream) - set(self.capture_layers))
        if outside or not self.capture_layers:
            raise ValueError(
                f"output_residual_stream names layer(s) {outside}, but under "
                "VLLM_LENS_CUDAGRAPH this server captures only the layers in "
                f"VLLM_LENS_CAPTURE_LAYERS: {list(self.capture_layers)}."
            )


@dataclass
class _SteerBuffers:
    """The steering buffers of one worker, ``[n_steered, max_tokens, hidden]`` each.
    ``add`` rows are added as they are; ``norm`` rows are unit rows for ``norm_match``."""

    add: torch.Tensor
    norm: torch.Tensor
    slots: dict[int, int]
    # Rows below this index can be nonzero; all other rows are zero.
    dirty: int = 0
    # The step's token positions and their row in the stacked requests, ``[2, max_tokens]``:
    # a pinned staging copy and its device copy, so the fill never waits for the stream.
    index_cpu: torch.Tensor | None = None
    index_gpu: torch.Tensor | None = None

    def rows(
        self, configs: list[SteeringVector]
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """The ``add`` and ``norm`` row of one request, ``(n_slots, hidden)`` each, with
        ``scale`` folded in; None when a vector is position-specific."""
        if any(cfg.activations.dim() != 2 for cfg in configs):
            return None
        add = torch.zeros(
            (self.add.shape[0], self.add.shape[2]),
            dtype=self.add.dtype,
            device=self.add.device,
        )
        norm = torch.zeros_like(add)
        for cfg in configs:
            for layer_idx, slot in self.slots.items():
                if layer_idx not in cfg.layer_index_map:
                    continue
                vec = cfg.activations[cfg.layer_index_map[layer_idx]].float()
                if cfg.norm_match:
                    # The op multiplies by the stream norm, so store v / ||v||.
                    vec = vec / (vec.norm() + 1e-6)
                    norm[slot] += (vec * cfg.scale).to(norm.dtype)
                else:
                    add[slot] += (vec * cfg.scale).to(add.dtype)
        return add, norm


# torch dispatches a custom op by name, so the op finds the buffers here.
# One worker process holds one model, so there is one entry.
_STEER_STATE: dict[str, _SteerBuffers] = {}


@torch.library.custom_op("vllm_lens::steer", mutates_args=("hidden_states",))
def _steer(
    hidden_states: torch.Tensor, residual: torch.Tensor | None, slot: int
) -> None:
    """Add this layer's buffer rows to the stream. Zero rows change nothing.
    The op has no branch, so it is traced into the graph with no graph boundary."""
    state = _STEER_STATE.get("buffers")
    if state is None:
        return
    n = hidden_states.shape[0]
    if n > state.add.shape[1]:
        raise RuntimeError(
            f"{n} tokens in one forward pass, but the steering buffers hold "
            f"max_num_batched_tokens = {state.add.shape[1]}."
        )
    # norm_match references the full stream, which for a fused-residual layer
    # is the sum of the two outputs.
    stream = hidden_states + residual if residual is not None else hidden_states
    stream_norm = torch.linalg.vector_norm(
        stream, dim=-1, keepdim=True, dtype=torch.float32
    )
    delta = state.add[slot, :n] + state.norm[slot, :n] * stream_norm
    hidden_states += delta.to(hidden_states.dtype)


@_steer.register_fake
def _steer_fake(
    hidden_states: torch.Tensor, residual: torch.Tensor | None, slot: int
) -> None:
    """Shape propagation only."""
    return None


def _fill_steer_buffers(extension: Any) -> None:
    """Write the steering rows of each request, before the forward pass reads them.
    Rows of the last step are zeroed first, so no stale row steers another request."""
    from vllm_lens._worker_ext import _apply_steering, _batch_layout, _steering_key

    state = _STEER_STATE["buffers"]
    state.add[:, : state.dirty].zero_()
    state.norm[:, : state.dirty].zero_()
    state.dirty = 0
    runner = extension.model_runner
    query_start_loc = _batch_layout(runner, warn=False)
    if query_start_loc is None or not extension._steering_data:
        return
    num_reqs = runner.input_batch.num_reqs
    # vLLM fills a pinned CPU copy of the layout before the forward pass, so
    # reading it needs no host sync; an older runner costs one sync per step.
    layout = getattr(runner, "query_start_loc", None)
    if layout is not None and hasattr(layout, "np"):
        bounds = layout.np[: num_reqs + 1].astype(np.int64)
    else:
        bounds = np.asarray(query_start_loc[: num_reqs + 1].tolist(), dtype=np.int64)

    # The 2D requests of this step: their rows and their token spans, gathered
    # into the buffers in one indexed copy per table after the loop.
    gathered: list[tuple[torch.Tensor, torch.Tensor]] = []
    spans: list[tuple[int, int]] = []
    for i in range(num_reqs):
        req_id = runner.input_batch.req_ids[i]
        req_state = runner.requests.get(req_id)
        if req_state is None or req_state.sampling_params is None:
            continue
        key = _steering_key(extension, req_id, req_state.sampling_params.extra_args)
        if key is None:
            continue
        start = int(bounds[i])
        end = int(bounds[i + 1])
        state.dirty = max(state.dirty, end)
        if key not in extension._steer_rows:
            try:
                extension._steer_rows[key] = state.rows(extension._steering_data[key])
            except Exception:
                # As in eager mode: a bad vector must not stop the engine.
                logger.warning("Steering failed for request %s", req_id, exc_info=True)
                extension._steer_rows[key] = state.rows([])
        rows = extension._steer_rows[key]
        if rows is not None:
            gathered.append(rows)
            spans.append((start, end))
            continue
        try:
            for cfg in extension._steering_data[key]:
                target = state.add
                if cfg.norm_match:
                    # The op multiplies by the stream norm, so store v / ||v||.
                    # Cached on the vector: this runs on every forward pass.
                    if not hasattr(cfg, "_unit_copy"):
                        unit = cfg.activations.float()
                        unit = unit / (unit.norm(dim=-1, keepdim=True) + 1e-6)
                        unit_copy = cfg.model_copy(
                            update={"activations": unit, "norm_match": False}
                        )
                        object.__setattr__(cfg, "_unit_copy", unit_copy)
                    cfg = cfg._unit_copy  # type: ignore[reportAttributeAccessIssue]
                    target = state.norm
                for layer_idx, slot in state.slots.items():
                    # _apply_steering adds to the rows, so vectors on one layer sum.
                    _apply_steering(
                        [cfg],
                        layer_idx,
                        target[slot],
                        start,
                        end,
                        req_state.num_computed_tokens,
                        target[slot],
                    )
        except Exception:
            logger.warning("Steering failed for request %s", req_id, exc_info=True)
            state.add[:, start:end].zero_()
            state.norm[:, start:end].zero_()

    if not gathered:
        return
    index = np.stack(
        [
            np.concatenate([np.arange(start, end) for start, end in spans]),
            np.repeat(np.arange(len(gathered)), [end - start for start, end in spans]),
        ]
    )
    n_tokens = index.shape[1]
    if state.index_cpu is None or state.index_gpu is None:
        index_gpu = torch.from_numpy(index).to(state.add.device)
    else:
        # A pageable copy would wait for every kernel queued before it.
        state.index_cpu[:, :n_tokens].numpy()[:] = index
        state.index_gpu[:, :n_tokens].copy_(
            state.index_cpu[:, :n_tokens], non_blocking=True
        )
        index_gpu = state.index_gpu[:, :n_tokens]
    positions = index_gpu[0]
    token_rows = index_gpu[1]
    # (n_tokens, n_slots, hidden) -> (n_slots, n_tokens, hidden)
    state.add[:, positions] = torch.stack([add for add, _ in gathered])[
        token_rows
    ].permute(1, 0, 2)
    state.norm[:, positions] = torch.stack([norm for _, norm in gathered])[
        token_rows
    ].permute(1, 0, 2)


def _install_op_hooks(worker: Any, op_args: dict[int, int], op: Any) -> None:
    """Give each layer in ``op_args`` a forward hook that calls ``op`` with its value."""
    from vllm_lens._worker_ext import _discover_layer_modules

    for layer_idx, layer in sorted(_discover_layer_modules(worker).items()):
        if layer_idx not in op_args:
            continue
        # A second load_model must not add a second hook.
        hooked = layer.__dict__.setdefault("_lens_ops", set())
        if str(op) in hooked:
            continue
        hooked.add(str(op))

        # Dynamo traces the hook body, so it is one op call and nothing else.
        def _hook(
            module: Any, args: Any, output: Any, op_arg: int = op_args[layer_idx]
        ) -> None:
            if isinstance(output, tuple):
                op(output[0], output[1], op_arg)
            else:
                op(output, None, op_arg)

        layer.register_forward_hook(_hook)


def _enable_buffer_steering(worker: Any, layers: tuple[int, ...]) -> None:
    """Allocate the steering buffers and hook the steer op onto ``layers``."""
    runner = worker.model_runner
    model = runner.model
    shape = (
        len(layers),
        runner.scheduler_config.max_num_batched_tokens,
        runner.model_config.get_hidden_size(),
    )
    dtype = runner.model_config.dtype
    device = next(model.parameters()).device
    slots = {layer: slot for slot, layer in enumerate(layers)}
    existing = _STEER_STATE.get("buffers")
    # Kept on a second load_model: a captured CUDA graph reads this address.
    if (
        existing is None
        or existing.slots != slots
        or existing.add.shape != shape
        or existing.add.dtype != dtype
        or existing.add.device != device
    ):
        pinned = device.type == "cuda"
        _STEER_STATE["buffers"] = _SteerBuffers(
            add=torch.zeros(shape, dtype=dtype, device=device),
            norm=torch.zeros(shape, dtype=dtype, device=device),
            slots=slots,
            index_cpu=torch.zeros((2, shape[1]), dtype=torch.int64, pin_memory=pinned),
            index_gpu=torch.zeros((2, shape[1]), dtype=torch.int64, device=device),
        )
    _install_op_hooks(worker, _STEER_STATE["buffers"].slots, torch.ops.vllm_lens.steer)
    megabytes = 2 * shape[0] * shape[1] * shape[2] * dtype.itemsize / 2**20
    logger.info(
        "Buffer steering enabled for layers %s, %.0f MB", list(layers), megabytes
    )


def _enable_aux_capture(worker: Any, layers: tuple[int, ...]) -> None:
    """Make the loaded model return ``layers`` as auxiliary hidden states."""
    runner = worker.model_runner
    model = runner.model
    # A second load_model finds the flag that the first call set here.
    set_here = getattr(runner, "_lens_aux_layers", None) is not None
    if runner.speculative_config is not None or (
        runner.use_aux_hidden_state_outputs and not set_here
    ):
        raise RuntimeError(
            "VLLM_LENS_CAPTURE_LAYERS cannot be used with speculative decoding: "
            "the draft model reads the same auxiliary hidden states."
        )
    if worker.parallel_config.pipeline_parallel_size > 1:
        raise RuntimeError(
            "VLLM_LENS_CAPTURE_LAYERS does not support pipeline parallelism: "
            "only the last stage returns auxiliary hidden states."
        )
    if not hasattr(model, "set_aux_hidden_state_layers"):
        raise RuntimeError(
            f"{type(model).__name__} does not implement "
            "set_aux_hidden_state_layers, so VLLM_LENS_CAPTURE_LAYERS cannot "
            "be served. Unset it to use the eager forward hooks."
        )
    # Aux index L + 1 is the residual stream after layer L; index 0 is the
    # embedding output. Set before warmup, because the trace freezes it.
    model.set_aux_hidden_state_layers(tuple(layer + 1 for layer in layers))
    # vLLM unpacks (hidden_states, aux_hidden_states) only under this flag.
    runner.use_aux_hidden_state_outputs = True
    runner._lens_aux_layers = layers
    logger.info("Aux capture enabled for layers %s", list(layers))


def install_graph_patches() -> None:
    """Patch the vLLM worker so an engine sets up its CUDA-graph settings at load."""
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        from vllm.v1.worker.gpu_worker import Worker
    except ImportError:
        logger.debug("No vLLM GPU worker; the CUDA-graph mode is not available.")
        return
    # register() can run twice in a process; a second wrap would capture twice.
    if getattr(Worker.load_model, "_lens_patched", False):
        return

    from vllm_lens._aux_models import install_gemma2_aux_hidden_states
    from vllm_lens._worker_ext import (
        _batch_layout,
        _capture_rows,
        _get_total_num_layers,
    )

    install_gemma2_aux_hidden_states()
    original_load_model = Worker.load_model
    original_model_forward = GPUModelRunner._model_forward

    def _load_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Load the model, then enable the paths this engine's config names."""
        result = original_load_model(self, *args, **kwargs)
        config = LensGraphConfig.from_vllm_config(self.vllm_config)
        named = (*config.capture_layers, *config.steer_layers)
        if not named:
            return result
        if not hasattr(self, "_reset_state"):
            raise RuntimeError(
                "The VLLM_LENS_*_LAYERS variables need the vllm-lens worker extension, "
                "but worker_extension_cls names another class."
            )
        n_layers = _get_total_num_layers(self)
        outside = [layer for layer in named if not 0 <= layer < n_layers]
        if outside:
            raise ValueError(
                f"The VLLM_LENS_*_LAYERS variables name layer(s) {outside}, but the "
                f"model has layers 0..{n_layers - 1}."
            )
        if config.capture_layers:
            _enable_aux_capture(self, config.capture_layers)
        if config.steer_layers:
            _enable_buffer_steering(self, config.steer_layers)
        self.model_runner._lens_extension = self
        self.model_runner._lens_graph_config = config
        # No request-level forward hooks: they cannot serve requests here.
        self._reset_state()
        self._hooks_installed = True
        return result

    def _model_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Fill the steering buffers, run the forward pass, store the captured rows."""
        extension = getattr(self, "_lens_extension", None)
        if extension is None:
            return original_model_forward(self, *args, **kwargs)
        if self._lens_graph_config.steer_layers:
            _fill_steer_buffers(extension)
        out = original_model_forward(self, *args, **kwargs)
        if (
            not extension._should_capture
            or not isinstance(out, tuple)
            or len(out) != 2
            or not isinstance(out[1], list)
        ):
            return out
        layers = self._lens_graph_config.capture_layers
        try:
            # A shorter list would label the layers wrongly, so store nothing.
            if len(out[1]) != len(layers):
                raise RuntimeError(
                    f"the model returned {len(out[1])} auxiliary hidden states for "
                    f"{len(layers)} capture layers"
                )
            query_start_loc = _batch_layout(self, warn=False)
            if query_start_loc is not None:
                # The model returns the aux states in layer order, as ``layers`` is.
                for layer_idx, hidden_states in zip(layers, out[1]):
                    _capture_rows(extension, layer_idx, hidden_states, query_start_loc)
        except Exception:
            logger.warning("Aux activation capture failed", exc_info=True)
        return out

    _load_model._lens_patched = True  # type: ignore[attr-defined]
    Worker.load_model = _load_model
    GPUModelRunner._model_forward = _model_forward

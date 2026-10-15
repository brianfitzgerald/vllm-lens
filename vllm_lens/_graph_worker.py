"""CUDA-graph mode on the worker: one op inside the graph steers and captures the named
layers through fixed buffers, which the patched forward fills before and reads after."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from vllm_lens._cudagraph import LensGraphConfig
from vllm_lens._worker_ext import (
    _apply_steering,
    _batch_layout,
    _capture_rows,
    _discover_layer_modules,
    _get_total_num_layers,
    _steered_requests,
)

logger = logging.getLogger(__name__)


@dataclass
class _GraphBuffers:
    """The fixed buffers of one worker, ``[n_layers, max_tokens, hidden]`` each: the steering
    ``add`` and ``norm`` rows of every token, and the captured residual stream."""

    add: torch.Tensor
    norm: torch.Tensor
    capture: torch.Tensor
    steer_slots: dict[int, int]
    capture_slots: dict[int, int]
    # The table row of every token, as a pinned staging copy and its device copy; the event
    # marks the last copy, so the next step does not overwrite it while it is queued.
    index_cpu: torch.Tensor
    index_gpu: torch.Tensor
    index_event: torch.cuda.Event | None
    # Steering rows below this token index can be nonzero.
    dirty: int = 0


# torch dispatches a custom op by name, so the op finds the buffers here.
# One worker process holds one model, so there is one entry.
_GRAPH_STATE: dict[str, _GraphBuffers] = {}


@torch.library.custom_op("vllm_lens::lens", mutates_args=("hidden_states",))
def _lens(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    steer_slot: int,
    capture_slot: int,
) -> None:
    """Add the layer's steering rows to the stream, then copy the stream to the capture
    buffer; a slot of -1 skips that part. Opaque to the compiler, so no graph boundary."""
    state = _GRAPH_STATE.get("buffers")
    if state is None:
        return
    n = hidden_states.shape[0]
    if n > state.capture.shape[1]:
        raise RuntimeError(
            f"{n} tokens in one forward pass, but the vllm-lens buffers hold "
            f"max_num_batched_tokens = {state.capture.shape[1]}."
        )
    if steer_slot >= 0:
        # norm_match references the full stream: for a fused-residual layer, the sum.
        stream = hidden_states + residual if residual is not None else hidden_states
        stream_norm = torch.linalg.vector_norm(
            stream, dim=-1, keepdim=True, dtype=torch.float32
        )
        delta = state.add[steer_slot, :n] + state.norm[steer_slot, :n] * stream_norm
        hidden_states += delta.to(hidden_states.dtype)
    if capture_slot >= 0:
        stream = hidden_states + residual if residual is not None else hidden_states
        state.capture[capture_slot, :n].copy_(stream)


@_lens.register_fake
def _lens_fake(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    steer_slot: int,
    capture_slot: int,
) -> None:
    """Shape propagation only."""
    return None


def _fill_steer_buffers(
    extension: Any, state: _GraphBuffers, query_start_loc: torch.Tensor | None
) -> None:
    """Write every token's steering rows before the forward pass: one gather per layer from
    the request table (row 0 steers nothing), then position-specific vectors per request."""
    state.add[:, : state.dirty].zero_()
    state.norm[:, : state.dirty].zero_()
    state.dirty = 0
    steered = _steered_requests(extension) if query_start_loc is not None else []
    if not steered:
        return
    runner = extension.model_runner
    num_reqs = runner.input_batch.num_reqs
    # The runner fills this pinned copy before the forward pass, so reading it needs no sync.
    bounds = runner.query_start_loc.np[: num_reqs + 1].astype(np.int64)
    n_tokens = int(bounds[-1])
    state.dirty = n_tokens
    rows = np.zeros(num_reqs, dtype=np.int64)
    for i, row, _ in steered:
        rows[i] = max(row, 0)
    table = extension._steer_table
    if rows.any():
        if state.index_event is not None:
            state.index_event.synchronize()
        state.index_cpu[:n_tokens].numpy()[:] = np.repeat(rows, np.diff(bounds))
        index = state.index_gpu[:n_tokens]
        index.copy_(state.index_cpu[:n_tokens], non_blocking=True)
        if state.index_event is not None:
            state.index_event.record()
        for layer, slot in state.steer_slots.items():
            if layer in table.add:
                state.add[slot, :n_tokens] = table.add[layer][index]
            if layer in table.norm:
                state.norm[slot, :n_tokens] = table.norm[layer][index]

    for i, row, key in steered:
        if row > 0:
            continue
        req_state = runner.requests[runner.input_batch.req_ids[i]]
        start, end = int(bounds[i]), int(bounds[i + 1])
        for cfg in extension._steering_data[key]:
            # The op multiplies the norm buffer by the stream norm, so it holds unit rows.
            target = state.norm if cfg.norm_match else state.add
            if cfg.norm_match:
                activations = cfg.activations.float()
                activations = activations / (
                    activations.norm(dim=-1, keepdim=True) + 1e-6
                )
                cfg = cfg.model_copy(
                    update={"activations": activations, "norm_match": False}
                )
            for layer, slot in state.steer_slots.items():
                # _apply_steering adds to the rows, so vectors on one layer sum.
                _apply_steering(
                    [cfg],
                    layer,
                    target[slot],
                    start,
                    end,
                    req_state.num_computed_tokens,
                    target[slot],
                )


def _enable_graph_buffers(worker: Any, config: LensGraphConfig) -> None:
    """Allocate the buffers, or keep them on a second load, and hook the op onto the
    named layers of this rank."""
    runner = worker.model_runner
    model = runner.model
    layer_map = _discover_layer_modules(worker)
    max_tokens = runner.scheduler_config.max_num_batched_tokens
    hidden = runner.model_config.get_hidden_size()
    dtype = runner.model_config.dtype
    device = next(model.parameters()).device
    steer_slots = {layer: slot for slot, layer in enumerate(config.steer_layers)}
    capture_slots = {
        layer: slot
        for slot, layer in enumerate(
            layer for layer in config.capture_layers if layer in layer_map
        )
    }
    shapes = (
        (len(steer_slots), max_tokens, hidden),
        (len(capture_slots), max_tokens, hidden),
    )
    existing = _GRAPH_STATE.get("buffers")
    # A captured CUDA graph reads these addresses, so a second load keeps them.
    if (
        existing is None
        or existing.steer_slots != steer_slots
        or existing.capture_slots != capture_slots
        or (existing.add.shape, existing.capture.shape) != shapes
        or existing.add.dtype != dtype
        or existing.add.device != device
    ):
        pinned = device.type == "cuda"
        _GRAPH_STATE["buffers"] = _GraphBuffers(
            add=torch.zeros(shapes[0], dtype=dtype, device=device),
            norm=torch.zeros(shapes[0], dtype=dtype, device=device),
            capture=torch.zeros(shapes[1], dtype=dtype, device=device),
            steer_slots=steer_slots,
            capture_slots=capture_slots,
            index_cpu=torch.zeros(max_tokens, dtype=torch.int64, pin_memory=pinned),
            index_gpu=torch.zeros(max_tokens, dtype=torch.int64, device=device),
            index_event=torch.cuda.Event() if pinned else None,
        )
    for layer_idx in sorted(set(steer_slots) | set(capture_slots)):
        layer = layer_map.get(layer_idx)
        # A second load must not add a second hook.
        if layer is None or layer.__dict__.get("_lens_op_hooked", False):
            continue
        layer.__dict__["_lens_op_hooked"] = True
        steer_slot = steer_slots.get(layer_idx, -1)
        capture_slot = capture_slots.get(layer_idx, -1)

        # Dynamo traces the hook body, so it is one op call and nothing else.
        def _hook(
            module: Any,
            args: Any,
            output: Any,
            steer_slot: int = steer_slot,
            capture_slot: int = capture_slot,
        ) -> None:
            if isinstance(output, tuple):
                torch.ops.vllm_lens.lens(output[0], output[1], steer_slot, capture_slot)
            else:
                torch.ops.vllm_lens.lens(output, None, steer_slot, capture_slot)

        layer.register_forward_hook(_hook)
    megabytes = (2 * shapes[0][0] + shapes[1][0]) * max_tokens * hidden
    logger.info(
        "CUDA-graph buffers: steer layers %s, capture layers %s, %.0f MB",
        list(steer_slots),
        list(capture_slots),
        megabytes * dtype.itemsize / 2**20,
    )


def install_graph_patches() -> None:
    """Patch the vLLM worker so an engine sets up its CUDA-graph settings at load."""
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

    # register() can run twice in a process; a second wrap would act twice.
    if getattr(Worker.load_model, "_lens_patched", False):
        return
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
        if self.parallel_config.pipeline_parallel_size > 1:
            raise RuntimeError(
                "The VLLM_LENS_*_LAYERS variables are not tested with pipeline "
                "parallelism. Unset them to use the eager forward hooks."
            )
        _enable_graph_buffers(self, config)
        self.model_runner._lens_extension = self
        # No request-level forward hooks: they cannot serve requests here.
        self._reset_state()
        self._hooks_installed = True
        return result

    def _model_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Fill the steering buffers, run the forward pass, store the captured rows."""
        extension = getattr(self, "_lens_extension", None)
        if extension is None:
            return original_model_forward(self, *args, **kwargs)
        state = _GRAPH_STATE["buffers"]
        query_start_loc = _batch_layout(self, warn=False)
        if state.steer_slots:
            _fill_steer_buffers(extension, state, query_start_loc)
        out = original_model_forward(self, *args, **kwargs)
        if query_start_loc is None or not extension._should_capture:
            return out
        try:
            for layer, slot in state.capture_slots.items():
                _capture_rows(extension, layer, state.capture[slot], query_start_loc)
        except Exception:
            logger.warning("vllm-lens graph capture failed", exc_info=True)
        return out

    _load_model._lens_patched = True  # type: ignore[attr-defined]
    Worker.load_model = _load_model
    GPUModelRunner._model_forward = _model_forward

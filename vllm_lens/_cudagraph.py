"""CUDA-graph mode: serve vllm-lens requests without forcing ``enforce_eager``.

Forward hooks cannot serve requests under ``torch.compile``: vLLM compiles once
and skips the guards, so the branch a hook takes during warmup is frozen in.
Capture uses the model's auxiliary hidden states, which are graph outputs.
Steering uses an op with no branch, which reads buffers the host fills.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)

# The key of these settings in ``VllmConfig.additional_config``.
CONFIG_KEY = "vllm_lens"


@dataclass(frozen=True)
class LensGraphConfig:
    """The CUDA-graph settings of one engine.

    Read from the environment when the engine config is created, and stored in
    ``VllmConfig.additional_config``. The front end and the workers of an engine
    then read the same values, and the values are part of the compile cache key.
    """

    enabled: bool = False
    capture_layers: tuple[int, ...] = ()
    steer_layers: tuple[int, ...] = ()

    @classmethod
    def from_env(cls) -> LensGraphConfig:
        """Read ``VLLM_LENS_CUDAGRAPH`` and the ``VLLM_LENS_*_LAYERS`` lists."""
        flag = os.environ.get("VLLM_LENS_CUDAGRAPH", "").strip().lower()
        layers = {
            name: tuple(
                sorted(
                    {
                        int(part)
                        for part in os.environ.get(name, "").split(",")
                        if part.strip()
                    }
                )
            )
            for name in ("VLLM_LENS_CAPTURE_LAYERS", "VLLM_LENS_STEER_LAYERS")
        }
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
        return cls(
            enabled=stored["enabled"],
            capture_layers=tuple(stored["capture_layers"]),
            steer_layers=tuple(stored["steer_layers"]),
        )

    def to_additional_config(self, existing: dict[str, Any] | None) -> dict[str, Any]:
        """``existing`` plus these settings, for ``EngineArgs.additional_config``."""
        return {**(existing or {}), CONFIG_KEY: asdict(self)}

    def reject_unserved(
        self,
        residual_stream: Any,
        steering_layers: set[int],
        needs_forward_hooks: bool,
    ) -> None:
        """Raise for a request that this CUDA-graph server cannot serve.

        ``residual_stream`` is the request's ``output_residual_stream`` value and
        ``steering_layers`` the layers its steering vectors name.
        """
        if not self.enabled:
            return
        if needs_forward_hooks:
            raise RuntimeError(
                "VLLM_LENS_CUDAGRAPH is set, so the vllm-lens forward hooks do not "
                "run: hooks are not available. Unset VLLM_LENS_CUDAGRAPH to serve "
                "this request in eager mode."
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
        wanted = residual_stream if isinstance(residual_stream, list) else []
        outside = sorted(set(wanted) - set(self.capture_layers))
        if outside or not self.capture_layers:
            raise ValueError(
                f"output_residual_stream names layer(s) {outside or 'all'}, but under "
                "VLLM_LENS_CUDAGRAPH this server captures only the layers in "
                f"VLLM_LENS_CAPTURE_LAYERS: {list(self.capture_layers)}."
            )


@dataclass
class _SteerBuffers:
    """The steering buffers of one worker, ``[n_steered, max_tokens, hidden]`` each.

    ``add`` holds the rows to add as they are. ``norm`` holds unit rows that the
    op scales by the norm of each token's residual stream (``norm_match``).
    """

    add: torch.Tensor
    norm: torch.Tensor
    slots: dict[int, int]
    # Rows below this index can be nonzero; all other rows are zero.
    dirty: int = 0


# torch dispatches a custom op by name, so the op finds the buffers here. A
# worker process holds one model, so this has at most one entry.
_STEER_STATE: dict[str, _SteerBuffers] = {}


@torch.library.custom_op("vllm_lens::steer", mutates_args=("hidden_states",))
def _steer(
    hidden_states: torch.Tensor, residual: torch.Tensor | None, slot: int
) -> None:
    """Add this layer's buffer rows to the stream. Zero rows change nothing.

    It has no branch on request state, so it is traced into the graph and
    costs no graph boundary. The host decides per request when it fills the rows.
    """
    state = _STEER_STATE.get("buffers")
    if state is None:
        return
    n = hidden_states.shape[0]
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

    Rows written in the last step are zeroed first: a stale row steers a
    request that did not ask for it.
    """
    from vllm_lens._worker_ext import (
        _abs_start,
        _apply_steering,
        _batch_layout,
        _find_steering_configs,
    )

    state = _STEER_STATE["buffers"]
    state.add[:, : state.dirty].zero_()
    state.norm[:, : state.dirty].zero_()
    state.dirty = 0
    runner = extension.model_runner
    layout = _batch_layout(runner)
    if layout is None:
        return
    query_start_loc, meta_with_qsl = layout

    for i in range(runner.input_batch.num_reqs):
        req_id = runner.input_batch.req_ids[i]
        req_state = runner.requests.get(req_id)
        if req_state is None or req_state.sampling_params is None:
            continue
        extra = req_state.sampling_params.extra_args
        configs = _find_steering_configs(extension, req_id, extra)
        if not configs:
            continue
        start = int(query_start_loc[i].item())
        end = int(query_start_loc[i + 1].item())
        abs_start = _abs_start(meta_with_qsl, i, end - start)
        state.dirty = max(state.dirty, end)
        for cfg in configs:
            target = state.add
            if cfg.norm_match:
                # The op multiplies by the stream norm, so store v / ||v||.
                unit = cfg.activations.float()
                unit = unit / (unit.norm(dim=-1, keepdim=True) + 1e-6)
                cfg = cfg.model_copy(update={"activations": unit, "norm_match": False})
                target = state.norm
            for layer_idx, slot in state.slots.items():
                # _apply_steering adds to the rows, so vectors on one layer sum.
                _apply_steering(
                    [cfg], layer_idx, target[slot], start, end, abs_start, target[slot]
                )


def _arm_aux_capture(worker: Any, layers: tuple[int, ...]) -> None:
    """Make the loaded model return ``layers`` as auxiliary hidden states."""
    model = worker.model_runner.model
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
    worker.model_runner.use_aux_hidden_state_outputs = True
    logger.info("aux capture armed for layers %s", list(layers))


def _arm_buffer_steering(worker: Any, layers: tuple[int, ...]) -> None:
    """Allocate the steering buffers and hook the steer op onto ``layers``."""
    from vllm.model_executor.models.utils import PPMissingLayer

    from vllm_lens._worker_ext import _get_layers

    runner = worker.model_runner
    model = runner.model
    # Allocated once: CUDA graph capture records the address every replay reads.
    shape = (
        len(layers),
        runner.scheduler_config.max_num_batched_tokens,
        runner.model_config.get_hidden_size(),
    )
    device = next(model.parameters()).device
    dtype = runner.model_config.dtype
    slots = {layer: slot for slot, layer in enumerate(layers)}
    _STEER_STATE["buffers"] = _SteerBuffers(
        add=torch.zeros(shape, dtype=dtype, device=device),
        norm=torch.zeros(shape, dtype=dtype, device=device),
        slots=slots,
    )
    for layer_idx, layer in enumerate(_get_layers(model)):
        if layer_idx not in slots or isinstance(layer, PPMissingLayer):
            continue

        # Dynamo traces the hook body, so it is one op call and nothing else.
        def _hook(
            module: Any, args: Any, output: Any, slot: int = slots[layer_idx]
        ) -> None:
            if isinstance(output, tuple):
                torch.ops.vllm_lens.steer(output[0], output[1], slot)
            else:
                torch.ops.vllm_lens.steer(output, None, slot)

        layer.register_forward_hook(_hook)
    megabytes = 2 * shape[0] * shape[1] * shape[2] * dtype.itemsize / 2**20
    logger.info("buffer steering armed for layers %s, %.0f MB", list(layers), megabytes)


def arm() -> None:
    """Patch the vLLM worker so an engine with armed layers sets them up at load."""
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

    from vllm_lens._worker_ext import _batch_layout, _capture_rows, _get_layers

    original_load_model = Worker.load_model
    original_model_forward = GPUModelRunner._model_forward

    def _load_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Load the model, then arm the paths this engine's config names."""
        result = original_load_model(self, *args, **kwargs)
        config = LensGraphConfig.from_vllm_config(self.vllm_config)
        if not config.capture_layers and not config.steer_layers:
            return result
        n_layers = len(_get_layers(self.model_runner.model))
        outside = [
            layer
            for layer in (*config.capture_layers, *config.steer_layers)
            if not 0 <= layer < n_layers
        ]
        if outside:
            raise ValueError(
                f"VLLM_LENS_CAPTURE_LAYERS / VLLM_LENS_STEER_LAYERS name layer(s) "
                f"{outside}, but the model has layers 0..{n_layers - 1}."
            )
        if config.capture_layers:
            _arm_aux_capture(self, config.capture_layers)
        if config.steer_layers:
            _arm_buffer_steering(self, config.steer_layers)
        self.model_runner._lens_extension = self
        self.model_runner._lens_graph_config = config
        # No request-level forward hooks: they cannot serve requests here.
        self._reset_state()
        self._hooks_installed = True
        return result

    def _model_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Fill the steering buffers, run the forward pass, store the captured rows."""
        extension = getattr(self, "_lens_extension", None)
        # A warmup run has no requests, and a D2H copy breaks graph capture.
        if extension is None or torch.cuda.is_current_stream_capturing():
            return original_model_forward(self, *args, **kwargs)
        if self._lens_graph_config.steer_layers:
            # Not caught: a failed fill would serve a steered request unsteered.
            _fill_steer_buffers(extension)
        out = original_model_forward(self, *args, **kwargs)
        if (
            not extension._should_capture
            or not isinstance(out, tuple)
            or len(out) != 2
            or not isinstance(out[1], list)
        ):
            return out
        try:
            layout = _batch_layout(self)
            if layout is not None:
                layers = self._lens_graph_config.capture_layers
                for layer_idx, hidden_states in zip(layers, out[1]):
                    _capture_rows(extension, layer_idx, hidden_states, layout[0])
        except Exception:
            logger.warning("aux activation capture failed", exc_info=True)
        return out

    Worker.load_model = _load_model
    GPUModelRunner._model_forward = _model_forward

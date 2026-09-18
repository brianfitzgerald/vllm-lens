"""CUDA-graph mode: serve vllm-lens requests without forcing ``enforce_eager``.

Forward hooks cannot serve requests under ``torch.compile``: vLLM compiles once
and skips the guards, so the branch a hook takes during warmup is frozen in.
Capture uses the model's auxiliary hidden states, which are graph outputs.
Steering uses an op with no branch, which reads buffers the host fills.
Hooks run in an op that is a piecewise split point, so its body runs eagerly.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm_lens._helpers.types import Hook


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
    hook_layers: tuple[int, ...] = ()

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
            for name in (
                "VLLM_LENS_CAPTURE_LAYERS",
                "VLLM_LENS_STEER_LAYERS",
                "VLLM_LENS_HOOK_LAYERS",
            )
        }
        hook_layers = layers["VLLM_LENS_HOOK_LAYERS"]
        # A hook layer already captures and steers, so a second path on the
        # same layer would capture or steer it twice.
        twice = sorted(
            set(hook_layers)
            & set(layers["VLLM_LENS_CAPTURE_LAYERS"] + layers["VLLM_LENS_STEER_LAYERS"])
        )
        if twice:
            raise ValueError(
                f"Layer(s) {twice} are in VLLM_LENS_HOOK_LAYERS and also in "
                "VLLM_LENS_CAPTURE_LAYERS or VLLM_LENS_STEER_LAYERS. A hook layer "
                "serves capture and steering, so name each layer once."
            )
        return cls(
            enabled=flag in ("1", "true", "yes", "on") or any(layers.values()),
            capture_layers=layers["VLLM_LENS_CAPTURE_LAYERS"],
            steer_layers=layers["VLLM_LENS_STEER_LAYERS"],
            hook_layers=hook_layers,
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
            hook_layers=tuple(sorted(stored.get("hook_layers", ()))),
        )

    def to_additional_config(self, existing: dict[str, Any] | None) -> dict[str, Any]:
        """``existing`` with these settings, or without them when the mode is off."""
        others = {
            key: value for key, value in (existing or {}).items() if key != CONFIG_KEY
        }
        return {**others, CONFIG_KEY: asdict(self)} if self.enabled else others

    def force_piecewise(self, engine_args: Any) -> None:
        """Set PIECEWISE on ``engine_args`` before vLLM derives its graph sizes.

        Under a full graph the whole model replays and the hook op body never runs.
        """
        if not self.hook_layers:
            return
        from vllm.config.compilation import CUDAGraphMode

        compilation = engine_args.compilation_config
        if compilation.cudagraph_mode not in (None, CUDAGraphMode.PIECEWISE):
            logger.warning(
                "VLLM_LENS_HOOK_LAYERS needs cudagraph_mode PIECEWISE, was %s",
                compilation.cudagraph_mode,
            )
        compilation.cudagraph_mode = CUDAGraphMode.PIECEWISE

    def split_at_hook_op(self, vllm_config: Any) -> None:
        """Make ``vllm_lens::hook`` a piecewise split point of ``vllm_config``."""
        if not self.hook_layers:
            return
        from vllm.config.compilation import CompilationMode

        compilation = vllm_config.compilation_config
        # vLLM asserts at engine start without the attention split, so say why.
        if (
            compilation.mode != CompilationMode.VLLM_COMPILE
            or not compilation.splitting_ops
        ):
            raise ValueError(
                "VLLM_LENS_HOOK_LAYERS needs piecewise compilation with the attention "
                "ops as split points. Remove --enforce-eager, a non-default "
                "compilation mode, attention fusion and sequence parallelism, or use "
                "VLLM_LENS_CAPTURE_LAYERS / VLLM_LENS_STEER_LAYERS."
            )
        # Append: vLLM has already put the attention ops here, and it does not
        # add them again when the list is set.
        if "vllm_lens::hook" not in compilation.splitting_ops:
            compilation.splitting_ops = [*compilation.splitting_ops, "vllm_lens::hook"]

    def reject_unserved(
        self,
        residual_stream: Any,
        steering_layers: set[int],
        hooks: list[Hook],
    ) -> None:
        """Raise for a request that this CUDA-graph server cannot serve.

        ``residual_stream`` is the request's ``output_residual_stream`` value,
        ``steering_layers`` the layers its steering vectors name, and ``hooks``
        its hooks (or the persistent hooks to register).
        """
        if not self.enabled:
            return
        if any(hook.pre for hook in hooks):
            raise ValueError(
                "Pre-hooks are not served under VLLM_LENS_CUDAGRAPH. Unset it to "
                "serve this request in eager mode."
            )
        if isinstance(residual_stream, str):
            try:
                residual_stream = json.loads(residual_stream)
            except (json.JSONDecodeError, ValueError):
                pass
        # "All layers" would return only the armed ones, with no layer labels.
        if residual_stream is not None and not isinstance(residual_stream, list):
            raise ValueError(
                "Under VLLM_LENS_CUDAGRAPH, output_residual_stream must be a list of "
                "layers from VLLM_LENS_CAPTURE_LAYERS or VLLM_LENS_HOOK_LAYERS: "
                f"{sorted({*self.capture_layers, *self.hook_layers})}."
            )
        if residual_stream is not None and not (
            self.capture_layers or self.hook_layers
        ):
            raise ValueError(
                "output_residual_stream is set, but under VLLM_LENS_CUDAGRAPH this "
                "server captures only the layers in VLLM_LENS_CAPTURE_LAYERS and "
                "VLLM_LENS_HOOK_LAYERS, which are empty."
            )
        wanted = {
            "output_residual_stream": (
                set(residual_stream or []),
                ("VLLM_LENS_CAPTURE_LAYERS", self.capture_layers),
            ),
            "apply_steering_vectors": (
                steering_layers,
                ("VLLM_LENS_STEER_LAYERS", self.steer_layers),
            ),
            "apply_hooks": (
                {layer for hook in hooks for layer in hook.layer_indices},
                ("VLLM_LENS_HOOK_LAYERS", ()),
            ),
        }
        for option, (layers, (variable, armed)) in wanted.items():
            outside = sorted(layers - set(armed) - set(self.hook_layers))
            if outside:
                raise ValueError(
                    f"{option} names layer(s) {outside}, but under VLLM_LENS_CUDAGRAPH "
                    f"this server serves it only on the layers in {variable}: "
                    f"{sorted({*armed, *self.hook_layers})}."
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
    assert n <= state.add.shape[1], "more tokens than max_num_batched_tokens"
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
        _apply_steering,
        _batch_layout,
        _find_steering_configs,
    )

    state = _STEER_STATE["buffers"]
    state.add[:, : state.dirty].zero_()
    state.norm[:, : state.dirty].zero_()
    state.dirty = 0
    runner = extension.model_runner
    query_start_loc = _batch_layout(runner, warn=False)
    if query_start_loc is None:
        return

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
        state.dirty = max(state.dirty, end)
        try:
            for cfg in configs:
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
            # As in eager mode: a bad vector must not stop the engine.
            logger.warning("steering failed for request %s", req_id, exc_info=True)
            state.add[:, start:end].zero_()
            state.norm[:, start:end].zero_()


def _install_op_hooks(model: Any, slots: dict[int, int], op: Any) -> None:
    """Give each layer in ``slots`` a forward hook that calls ``op`` with its slot."""
    from vllm.model_executor.models.utils import PPMissingLayer

    from vllm_lens._worker_ext import _get_layers

    for layer_idx, layer in enumerate(_get_layers(model)):
        if layer_idx not in slots or isinstance(layer, PPMissingLayer):
            continue
        # A second load_model must not add a second hook.
        hooked = layer.__dict__.setdefault("_lens_ops", set())
        if str(op) in hooked:
            continue
        hooked.add(str(op))

        # Dynamo traces the hook body, so it is one op call and nothing else.
        def _hook(
            module: Any, args: Any, output: Any, slot: int = slots[layer_idx]
        ) -> None:
            if isinstance(output, tuple):
                op(output[0], output[1], slot)
            else:
                op(output, None, slot)

        layer.register_forward_hook(_hook)


# The worker of this process, for the hook op, which torch dispatches by name.
_HOOK_STATE: dict[str, Any] = {}


@torch.library.custom_op("vllm_lens::hook", mutates_args=("hidden_states",))
def _split_hook(
    hidden_states: torch.Tensor, residual: torch.Tensor | None, layer_idx: int
) -> None:
    """Run ``_hook_inner`` for one layer and write its result into ``hidden_states``.

    The op is in ``splitting_ops``, so vLLM calls this body as plain Python
    between two graph pieces, and it can branch on the requests of the batch.
    """
    from vllm_lens._worker_ext import _hook_inner

    extension = _HOOK_STATE.get("extension")
    if extension is None:
        return
    output = hidden_states if residual is None else (hidden_states, residual)
    try:
        modified = _hook_inner(extension, layer_idx, output)
    except Exception:
        # As the eager forward hook does: log, and serve the step unchanged.
        logger.warning("split hook failed on layer %s", layer_idx, exc_info=True)
        return
    if modified is not None:
        hidden_states.copy_(modified[0] if isinstance(modified, tuple) else modified)


@_split_hook.register_fake
def _split_hook_fake(
    hidden_states: torch.Tensor, residual: torch.Tensor | None, layer_idx: int
) -> None:
    """Shape propagation only."""
    return None


def _arm_buffer_steering(worker: Any, layers: tuple[int, ...]) -> None:
    """Allocate the steering buffers and hook the steer op onto ``layers``."""
    runner = worker.model_runner
    model = runner.model
    shape = (
        len(layers),
        runner.scheduler_config.max_num_batched_tokens,
        runner.model_config.get_hidden_size(),
    )
    dtype = runner.model_config.dtype
    existing = _STEER_STATE.get("buffers")
    # Allocated once: CUDA graph capture records the address every replay reads.
    if existing is None or existing.add.shape != shape or existing.add.dtype != dtype:
        device = next(model.parameters()).device
        _STEER_STATE["buffers"] = _SteerBuffers(
            add=torch.zeros(shape, dtype=dtype, device=device),
            norm=torch.zeros(shape, dtype=dtype, device=device),
            slots={layer: slot for slot, layer in enumerate(layers)},
        )
    _install_op_hooks(model, _STEER_STATE["buffers"].slots, torch.ops.vllm_lens.steer)
    megabytes = 2 * shape[0] * shape[1] * shape[2] * dtype.itemsize / 2**20
    logger.info("buffer steering armed for layers %s, %.0f MB", list(layers), megabytes)


def _arm_aux_capture(worker: Any, layers: tuple[int, ...]) -> None:
    """Make the loaded model return ``layers`` as auxiliary hidden states."""
    from packaging.version import Version
    from vllm import __version__ as vllm_version

    runner = worker.model_runner
    model = runner.model
    if Version(vllm_version) < Version("0.18"):
        raise RuntimeError(
            "VLLM_LENS_CAPTURE_LAYERS needs vllm >= 0.18: earlier versions number "
            "the auxiliary hidden states differently."
        )
    if runner.speculative_config is not None or runner.use_aux_hidden_state_outputs:
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
    logger.info("aux capture armed for layers %s", list(layers))


def arm() -> None:
    """Patch the vLLM worker so an engine with armed layers sets them up at load."""
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        from vllm.v1.worker.gpu_worker import Worker
    except ImportError:
        logger.warning("No vLLM GPU worker; the CUDA-graph mode is not available.")
        return
    # register() can run twice in a process; a second wrap would capture twice.
    if getattr(Worker.load_model, "_lens_armed", False):
        return

    from vllm_lens._worker_ext import _batch_layout, _capture_rows, _get_layers

    original_load_model = Worker.load_model
    original_model_forward = GPUModelRunner._model_forward

    def _load_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Load the model, then arm the paths this engine's config names."""
        result = original_load_model(self, *args, **kwargs)
        config = LensGraphConfig.from_vllm_config(self.vllm_config)
        armed = (*config.capture_layers, *config.steer_layers, *config.hook_layers)
        if not armed:
            return result
        if not hasattr(self, "_reset_state"):
            raise RuntimeError(
                "The VLLM_LENS_*_LAYERS variables need the vllm-lens worker extension, "
                "but worker_extension_cls names another class."
            )
        n_layers = len(_get_layers(self.model_runner.model))
        outside = [layer for layer in armed if not 0 <= layer < n_layers]
        if outside:
            raise ValueError(
                f"The VLLM_LENS_*_LAYERS variables name layer(s) {outside}, but the "
                f"model has layers 0..{n_layers - 1}."
            )
        if config.capture_layers:
            _arm_aux_capture(self, config.capture_layers)
        if config.steer_layers:
            _arm_buffer_steering(self, config.steer_layers)
        if config.hook_layers:
            _HOOK_STATE["extension"] = self
            _install_op_hooks(
                self.model_runner.model,
                {layer: layer for layer in config.hook_layers},
                torch.ops.vllm_lens.hook,
            )
            logger.info("split hook armed for layers %s", list(config.hook_layers))
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
                for layer_idx, hidden_states in zip(layers, out[1]):
                    _capture_rows(extension, layer_idx, hidden_states, query_start_loc)
        except Exception:
            logger.warning("aux activation capture failed", exc_info=True)
        return out

    _load_model._lens_armed = True  # type: ignore[attr-defined]
    Worker.load_model = _load_model
    GPUModelRunner._model_forward = _model_forward

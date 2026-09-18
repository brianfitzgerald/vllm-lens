"""CUDA-graph mode: serve vllm-lens requests without forcing ``enforce_eager``.

Forward hooks cannot serve requests under ``torch.compile``: vLLM compiles once
and skips the guards, so the branch a hook takes during warmup is frozen in.
Capture uses the model's auxiliary hidden states, which are graph outputs.
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

    @classmethod
    def from_env(cls) -> LensGraphConfig:
        """Read ``VLLM_LENS_CUDAGRAPH`` and ``VLLM_LENS_CAPTURE_LAYERS``."""
        flag = os.environ.get("VLLM_LENS_CUDAGRAPH", "").strip().lower()
        raw_layers = os.environ.get("VLLM_LENS_CAPTURE_LAYERS", "")
        capture_layers = tuple(
            sorted({int(part) for part in raw_layers.split(",") if part.strip()})
        )
        return cls(
            enabled=flag in ("1", "true", "yes", "on") or bool(capture_layers),
            capture_layers=capture_layers,
        )

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> LensGraphConfig:
        """Read the settings that ``to_additional_config`` stored."""
        stored = (getattr(vllm_config, "additional_config", None) or {}).get(CONFIG_KEY)
        if not isinstance(stored, dict):
            return cls()
        return cls(
            enabled=stored["enabled"], capture_layers=tuple(stored["capture_layers"])
        )

    def to_additional_config(self, existing: dict[str, Any] | None) -> dict[str, Any]:
        """``existing`` plus these settings, for ``EngineArgs.additional_config``."""
        return {**(existing or {}), CONFIG_KEY: asdict(self)}

    def reject_unserved(self, residual_stream: Any, needs_forward_hooks: bool) -> None:
        """Raise for a request that this CUDA-graph server cannot serve.

        ``residual_stream`` is the request's ``output_residual_stream`` value.
        """
        if not self.enabled:
            return
        if needs_forward_hooks:
            raise RuntimeError(
                "VLLM_LENS_CUDAGRAPH is set, so the vllm-lens forward hooks do not "
                "run: steering and hooks are not available. Unset "
                "VLLM_LENS_CUDAGRAPH to serve this request in eager mode."
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
        if not config.capture_layers:
            return result
        n_layers = len(_get_layers(self.model_runner.model))
        outside = [
            layer for layer in config.capture_layers if not 0 <= layer < n_layers
        ]
        if outside:
            raise ValueError(
                f"VLLM_LENS_CAPTURE_LAYERS names layer(s) {outside}, but the model "
                f"has layers 0..{n_layers - 1}."
            )
        _arm_aux_capture(self, config.capture_layers)
        self.model_runner._lens_extension = self
        self.model_runner._lens_graph_config = config
        # No request-level forward hooks: they cannot serve requests here.
        self._reset_state()
        self._hooks_installed = True
        return result

    def _model_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Run the forward pass, then store the rows the requests asked for."""
        out = original_model_forward(self, *args, **kwargs)
        extension = getattr(self, "_lens_extension", None)
        if (
            extension is None
            or not extension._should_capture
            or not isinstance(out, tuple)
            or len(out) != 2
            or not isinstance(out[1], list)
            # A warmup run has no requests, and a D2H copy breaks graph capture.
            or torch.cuda.is_current_stream_capturing()
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

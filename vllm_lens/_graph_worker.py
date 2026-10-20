"""CUDA-graph mode on the worker: ops inside the graph steer each named layer from the
token buffers and copy its stream into a fixed buffer, read after the pass."""

from __future__ import annotations

import logging
from typing import Any

import torch
from vllm import envs

from vllm_lens._cudagraph import LensGraphConfig
from vllm_lens._worker_ext import (
    _batch_layout,
    _capture_rows,
    _discover_layer_modules,
    _steer,
    _steering_buffers,
)

logger = logging.getLogger("vllm.lens")


@torch.library.custom_op("vllm_lens::capture", mutates_args=("out",))
def _capture(
    hidden_states: torch.Tensor, residual: torch.Tensor | None, out: torch.Tensor
) -> None:
    """Copy each token's residual stream (``hidden + residual``) into ``out``."""
    n = hidden_states.shape[0]
    out[:n].copy_(hidden_states if residual is None else hidden_states + residual)


@torch.library.custom_op("vllm_lens::steer", mutates_args=("hidden_states",))
def _steer_op(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    add: torch.Tensor,
    norm: torch.Tensor,
) -> None:
    """Apply the token buffers to the stream in place."""
    n = hidden_states.shape[0]
    _steer(hidden_states, residual, add[:n], norm[:n])


def _steer_hook(module: Any, args: Any, output: Any) -> None:
    """Call the steer op on the layer output; Dynamo traces this body into the graph."""
    hidden, residual = output if isinstance(output, tuple) else (output, None)
    torch.ops.vllm_lens.steer(hidden, residual, module._lens_add, module._lens_norm)


def _capture_hook(module: Any, args: Any, output: Any) -> None:
    """Call the capture op on the layer output; Dynamo traces this body into the graph."""
    hidden, residual = output if isinstance(output, tuple) else (output, None)
    torch.ops.vllm_lens.capture(hidden, residual, module._lens_capture)


def install_graph_patches() -> None:
    """Patch the vLLM worker so an engine sets up its CUDA-graph mode at load."""
    from vllm.v1.worker.gpu_worker import Worker

    original_load_model = Worker.load_model

    def _load_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Load the model, then set up the layers this engine's config names."""
        result = original_load_model(self, *args, **kwargs)
        config = LensGraphConfig.from_vllm_config(self.vllm_config)
        if not config.enabled:
            return result
        # Each of these splits or pads a forward pass in a way the buffers do not follow.
        refused = {
            "pipeline parallelism": self.parallel_config.pipeline_parallel_size > 1,
            "micro-batching (DBO)": self.parallel_config.use_ubatching,
            "sequence parallelism": bool(
                self.vllm_config.compilation_config.pass_config.enable_sp
            ),
            # vLLM turns this on for some architectures; it runs no torch.compile.
            "breakable CUDA graphs": envs.VLLM_USE_BREAKABLE_CUDAGRAPH,
        }
        for name, used in refused.items():
            if used:
                raise RuntimeError(
                    f"The CUDA-graph mode does not support {name}. "
                    "Remove additional_config['vllm_lens'] to use the eager forward hooks."
                )
        layer_map = _discover_layer_modules(self)
        missing = sorted(
            {*config.capture_layers, *config.steer_layers} - layer_map.keys()
        )
        if missing:
            raise ValueError(
                f"additional_config['vllm_lens'] names layer(s) {missing}, but the "
                f"model has layers 0..{len(layer_map) - 1}."
            )
        runner = self.model_runner
        max_tokens = runner.scheduler_config.max_num_batched_tokens
        hidden = runner.model_config.get_hidden_size()
        steering: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        buffers: dict[int, torch.Tensor] = {}
        model_forward = runner._model_forward
        warned = False
        filled: dict[int, int] = {}

        def _model_forward(*args: Any, **kwargs: Any) -> Any:
            """Fill steering buffers, run the model, then store captured rows."""
            nonlocal warned
            for layer_idx, (add, norm) in steering.items():
                plan = self._steer_pass
                spans = plan.get(layer_idx, []) if plan is not None else []
                length = spans[-1].end if spans else 0
                if length:
                    _steering_buffers(spans, length, add[:length], norm[:length])
                previous = filled.get(layer_idx, 0)
                if previous > length:
                    add[length:previous].zero_()
                    norm[length:previous].zero_()
                filled[layer_idx] = length
            out = model_forward(*args, **kwargs)
            query_start_loc = _batch_layout(runner, warn=False)
            if query_start_loc is None:
                return out
            try:
                for layer_idx, buffer in buffers.items():
                    _capture_rows(self, layer_idx, buffer, query_start_loc)
            except Exception:
                # Only the first failure is logged.
                if not warned:
                    warned = True
                    logger.warning("vllm-lens graph capture failed", exc_info=True)
            return out

        runner._model_forward = _model_forward
        # The planner wraps this so request slices are ready before the model runs.
        # The flag keeps install_hooks from adding the eager hooks.
        self._reset_state()
        self._hooks_installed = True
        for layer_idx in config.steer_layers:
            # Graph replay reads buffers with fixed addresses.
            add = torch.zeros(
                (max_tokens, hidden), dtype=torch.float32, device=self.device
            )
            norm = torch.zeros_like(add)
            steering[layer_idx] = (add, norm)
            layer = layer_map[layer_idx]
            layer.register_buffer("_lens_add", add, persistent=False)
            layer.register_buffer("_lens_norm", norm, persistent=False)
            # Registered first, so a layer steers before it captures.
            layer.register_forward_hook(_steer_hook)
        # Only the rank that captures needs a capture buffer.
        for layer_idx in config.capture_layers if self._should_capture else ():
            buffers[layer_idx] = torch.zeros(
                (max_tokens, hidden),
                dtype=runner.model_config.dtype,
                device=self.device,
            )
            layer_map[layer_idx].register_buffer(
                "_lens_capture", buffers[layer_idx], persistent=False
            )
            layer_map[layer_idx].register_forward_hook(_capture_hook)
        logger.info(
            "CUDA-graph mode: steer layers %s, capture layers %s",
            list(config.steer_layers),
            list(buffers),
        )
        return result

    Worker.load_model = _load_model

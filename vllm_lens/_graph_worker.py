"""CUDA-graph mode on the worker: ops inside the graph steer each named layer from the
token buffers and copy its stream into a fixed buffer, read after the pass."""

from __future__ import annotations

import logging
from typing import Any

import torch
from vllm import envs
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.worker.gpu_worker import Worker

from vllm_lens._cudagraph import LensGraphConfig
from vllm_lens._worker_ext import (
    HiddenStatesExtension,
    _batch_layout,
    _capture_rows,
    _discover_layer_modules,
    _steer,
    _steering_buffers,
)

logger = logging.getLogger("vllm.lens")

# The ops live as long as this library, which is why it is held at module level.
_LIBRARY = torch.library.Library("vllm_lens", "FRAGMENT")


def _capture_op(
    hidden_states: torch.Tensor, residual: torch.Tensor | None, out: torch.Tensor
) -> None:
    """Copy each token's residual stream (``hidden + residual``) into ``out``."""
    n = hidden_states.shape[0]
    out[:n].copy_(hidden_states if residual is None else hidden_states + residual)


def _steer_op(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    add: torch.Tensor,
    norm: torch.Tensor | None,
) -> None:
    """Apply the token buffers to the stream in place; with no ``norm``, only add."""
    n = hidden_states.shape[0]
    _steer(hidden_states, residual, add[:n], None if norm is None else norm[:n])


direct_register_custom_op(
    "capture", _capture_op, mutates_args=["out"], target_lib=_LIBRARY
)
direct_register_custom_op(
    "steer", _steer_op, mutates_args=["hidden_states"], target_lib=_LIBRARY
)


def _steer_hook(module: Any, args: Any, output: Any) -> None:
    """Call the steer op on the layer output; Dynamo traces this body into the graph."""
    hidden, residual = output if isinstance(output, tuple) else (output, None)
    torch.ops.vllm_lens.steer(hidden, residual, module._lens_add, module._lens_norm)


def _capture_hook(module: Any, args: Any, output: Any) -> None:
    """Call the capture op on the layer output; Dynamo traces this body into the graph."""
    hidden, residual = output if isinstance(output, tuple) else (output, None)
    torch.ops.vllm_lens.capture(hidden, residual, module._lens_capture)


class GraphWorker(Worker, HiddenStatesExtension):
    """The vLLM worker of an engine with layer lists: sets up the ops when the model loads."""

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        """Load the model, then set up the layers this engine's config names."""
        super().load_model(load_dummy_weights=load_dummy_weights)
        config = LensGraphConfig.from_vllm_config(self.vllm_config)
        compilation = self.vllm_config.compilation_config
        # Each of these splits or pads a forward pass in a way the buffers do not follow.
        refused = {
            "pipeline parallelism": self.parallel_config.pipeline_parallel_size > 1,
            "micro-batching (DBO)": self.parallel_config.use_ubatching,
            "sequence parallelism": bool(compilation.pass_config.enable_sp),
            "Inductor graph partition": compilation.use_inductor_graph_partition,
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
        steering: dict[int, tuple[torch.Tensor, torch.Tensor | None]] = {}
        buffers: dict[int, torch.Tensor] = {}
        filled: dict[int, int] = {}
        model_forward = runner._model_forward

        def _model_forward(*args: Any, **kwargs: Any) -> Any:
            """Fill the steering buffers, run the model, then store the captured rows."""
            plan = self._steer_pass or {}
            for layer_idx, (add, norm) in steering.items():
                spans = plan.get(layer_idx, [])
                length = spans[-1].end if spans else 0
                if spans:
                    _steering_buffers(
                        spans,
                        length,
                        add[:length],
                        None if norm is None else norm[:length],
                    )
                previous = filled.get(layer_idx, 0)
                if previous > length:
                    add[length:previous].zero_()
                    if norm is not None:
                        norm[length:previous].zero_()
                filled[layer_idx] = length
            out = model_forward(*args, **kwargs)
            query_start_loc = _batch_layout(runner, warn=False)
            if query_start_loc is not None:
                for layer_idx, buffer in buffers.items():
                    _capture_rows(self, layer_idx, buffer, query_start_loc)
            return out

        runner._model_forward = _model_forward
        # The planner wraps this so request slices are ready before the model runs.
        # The flag keeps install_hooks from adding the eager hooks.
        self._init_state()
        self._hooks_installed = True
        for layer_idx in config.steer_layers:
            # Graph replay reads buffers with fixed addresses.
            add = torch.zeros(
                (max_tokens, hidden), dtype=torch.float32, device=self.device
            )
            norm = torch.zeros_like(add) if config.steer_norm_match else None
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
            "CUDA-graph mode: steer layers %s (norm_match %s), capture layers %s",
            list(config.steer_layers),
            config.steer_norm_match,
            list(buffers),
        )

"""CUDA-graph mode on the worker: ops inside the graph steer each named layer from the
steering row table and copy its stream into a fixed buffer, read after the pass."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

from vllm_lens._cudagraph import LensGraphConfig
from vllm_lens._worker_ext import (
    _batch_layout,
    _capture_rows,
    _discover_layer_modules,
    _get_total_num_layers,
    _steer,
    _SteerTable,
)

logger = logging.getLogger("vllm.lens")


@torch.library.custom_op("vllm_lens::capture", mutates_args=("out",))
def _capture(
    hidden_states: torch.Tensor, residual: torch.Tensor | None, out: torch.Tensor
) -> None:
    """Copy each token's residual stream (``hidden + residual``) into ``out``."""
    n = hidden_states.shape[0]
    if n > out.shape[0]:
        raise RuntimeError(
            f"{n} tokens in one forward pass, but the vllm-lens buffers hold "
            f"max_num_batched_tokens = {out.shape[0]}."
        )
    out[:n].copy_(hidden_states if residual is None else hidden_states + residual)


@_capture.register_fake
def _capture_fake(
    hidden_states: torch.Tensor, residual: torch.Tensor | None, out: torch.Tensor
) -> None:
    """Shape propagation only."""
    return None


@torch.library.custom_op("vllm_lens::steer", mutates_args=("hidden_states",))
def _steer_op(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    add: torch.Tensor,
    norm: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    """Add each token's table row to the stream in place; row 0 steers nothing."""
    _steer(hidden_states, residual, add, norm, rows)


@_steer_op.register_fake
def _steer_op_fake(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    add: torch.Tensor,
    norm: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    """Shape propagation only."""
    return None


def _steer_hook(module: Any, args: Any, output: Any) -> None:
    """Call the steer op on the layer output; Dynamo traces this body into the graph."""
    hidden, residual = output if isinstance(output, tuple) else (output, None)
    torch.ops.vllm_lens.steer(
        hidden, residual, module._lens_add, module._lens_norm, module._lens_rows
    )


def _capture_hook(module: Any, args: Any, output: Any) -> None:
    """Call the capture op on the layer output; Dynamo traces this body into the graph."""
    if isinstance(output, tuple):
        torch.ops.vllm_lens.capture(output[0], output[1], module._lens_capture)
    else:
        torch.ops.vllm_lens.capture(output, None, module._lens_capture)


def install_graph_patches() -> None:
    """Patch the vLLM worker so an engine sets up its CUDA-graph mode at load."""
    from vllm.v1.worker.gpu_worker import Worker

    # register() can run twice in a process; a second wrap would act twice.
    if getattr(Worker.load_model, "_lens_patched", False):
        return
    original_load_model = Worker.load_model

    def _load_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Load the model, then set up the layers this engine's config names."""
        result = original_load_model(self, *args, **kwargs)
        config = LensGraphConfig.from_vllm_config(self.vllm_config)
        if not config.enabled:
            return result
        if not hasattr(self, "_reset_state"):
            raise RuntimeError(
                "additional_config['vllm_lens'] needs the vllm-lens worker extension, "
                "but worker_extension_cls names another class."
            )
        n_layers = _get_total_num_layers(self)
        named = (*config.capture_layers, *config.steer_layers)
        outside = [layer for layer in named if not 0 <= layer < n_layers]
        if outside:
            raise ValueError(
                f"additional_config['vllm_lens'] names layer(s) {outside}, but the "
                f"model has layers 0..{n_layers - 1}."
            )
        # Each of these splits or pads a forward pass in a way the buffers do not follow.
        refused = {
            "pipeline parallelism": self.parallel_config.pipeline_parallel_size > 1,
            "micro-batching (DBO)": self.parallel_config.use_ubatching,
            "sequence parallelism": bool(
                self.vllm_config.compilation_config.pass_config.enable_sp
            ),
            # vLLM turns this on for some architectures; it runs no torch.compile.
            "breakable CUDA graphs": os.environ.get("VLLM_USE_BREAKABLE_CUDAGRAPH")
            == "1",
        }
        for name, used in refused.items():
            if used:
                raise RuntimeError(
                    f"The CUDA-graph mode does not support {name}. "
                    "Remove additional_config['vllm_lens'] to use the eager forward hooks."
                )
        # No request-level forward hooks: they cannot serve requests here.
        self._reset_state()
        self._hooks_installed = True
        runner = self.model_runner
        layer_map = _discover_layer_modules(self)
        # Only the rank that captures needs a capture buffer.
        capture_layers = config.capture_layers if self._should_capture else ()
        max_tokens = runner.scheduler_config.max_num_batched_tokens
        hidden = runner.model_config.get_hidden_size()
        dtype = runner.model_config.dtype
        device = next(runner.model.parameters()).device
        if config.steer_layers:
            # The op reads the eager table, allocated once so its address stays fixed.
            table = self._steer_table = _SteerTable(
                device=device,
                hidden=hidden,
                n_rows=1 + runner.scheduler_config.max_num_seqs,
            )
            rows = torch.zeros(max_tokens, dtype=torch.int32, device=device)
            for layer_idx in config.steer_layers:
                add, norm = table.layer(layer_idx)
                layer = layer_map[layer_idx]
                layer.register_buffer("_lens_add", add, persistent=False)
                layer.register_buffer("_lens_norm", norm, persistent=False)
                layer.register_buffer("_lens_rows", rows, persistent=False)
                # Registered first, so a layer steers before it captures.
                layer.register_forward_hook(_steer_hook)

            def _fill_rows(plan: Any) -> None:
                """Write each token's table row where the op reads it; padding takes row 0."""
                # A graph capture must not record this write into the graph.
                if torch.cuda.is_current_stream_capturing():
                    return
                rows.zero_()
                if plan is not None:
                    rows[: plan.rows.shape[0]].copy_(plan.rows)

            self._on_steer_pass = _fill_rows
        buffers = {
            layer_idx: torch.zeros((max_tokens, hidden), dtype=dtype, device=device)
            for layer_idx in capture_layers
        }
        for layer_idx, buffer in buffers.items():
            layer_map[layer_idx].register_buffer(
                "_lens_capture", buffer, persistent=False
            )
            layer_map[layer_idx].register_forward_hook(_capture_hook)
        logger.info(
            "CUDA-graph mode: steer layers %s, capture layers %s, %.0f MB of capture buffers",
            list(config.steer_layers),
            list(capture_layers),
            len(capture_layers) * max_tokens * hidden * dtype.itemsize / 2**20,
        )
        model_forward = runner._model_forward
        warned = False

        def _model_forward(*args: Any, **kwargs: Any) -> Any:
            """Run the forward pass, then store the captured rows."""
            nonlocal warned
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
        return result

    _load_model._lens_patched = True  # type: ignore[attr-defined]
    Worker.load_model = _load_model

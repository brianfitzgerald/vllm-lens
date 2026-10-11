"""Auxiliary hidden states for vLLM models that lack the EAGLE-3 interface.
``VLLM_LENS_CAPTURE_LAYERS`` reads the captured layers through that interface."""

from __future__ import annotations

import logging
from itertools import islice
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _gemma2_forward(
    self: Any,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: Any | None,
    inputs_embeds: torch.Tensor | None = None,
) -> Any:
    """``Gemma2Model.forward`` that also returns the residual stream after each layer
    in ``aux_hidden_state_layers``, as ``Qwen2Model.forward`` does."""
    from vllm.distributed import get_pp_group
    from vllm.sequence import IntermediateTensors

    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]
    aux_hidden_states: list[torch.Tensor] = []
    # Index 0 is the embedding output; index L + 1 is the residual after layer L.
    if self.start_layer in self.aux_hidden_state_layers and residual is None:
        aux_hidden_states.append(hidden_states)
    for idx, layer in enumerate(
        islice(self.layers, self.start_layer, self.end_layer), start=self.start_layer
    ):
        hidden_states, residual = layer(positions, hidden_states, residual)
        if idx + 1 in self.aux_hidden_state_layers:
            aux_hidden_states.append(hidden_states + residual)
    if not get_pp_group().is_last_rank:
        return IntermediateTensors(
            {"hidden_states": hidden_states, "residual": residual}  # type: ignore[reportArgumentType]
        )
    hidden_states, _ = self.norm(hidden_states, residual)
    if aux_hidden_states:
        return hidden_states, aux_hidden_states
    return hidden_states


def _set_aux_hidden_state_layers(self: Any, layers: tuple[int, ...]) -> None:
    """Name the aux indices the text model returns; the EAGLE-3 interface's method."""
    self.model.aux_hidden_state_layers = layers


def install_gemma2_aux_hidden_states() -> None:
    """Give Gemma2 ``set_aux_hidden_state_layers`` before any model is built.
    ``torch.compile`` binds ``forward`` when the model is built, so this runs at register."""
    try:
        from vllm.model_executor.models.gemma2 import Gemma2ForCausalLM, Gemma2Model
    except ImportError:
        return
    if hasattr(Gemma2ForCausalLM, "set_aux_hidden_state_layers"):
        return
    setattr(Gemma2Model, "aux_hidden_state_layers", ())
    Gemma2Model.forward = _gemma2_forward
    setattr(
        Gemma2ForCausalLM, "set_aux_hidden_state_layers", _set_aux_hidden_state_layers
    )
    logger.debug("Gemma2 returns auxiliary hidden states for VLLM_LENS_CAPTURE_LAYERS")

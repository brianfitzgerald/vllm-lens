"""Discovery of the decoder layers of the model a worker serves."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
from vllm.model_executor.models.utils import PPMissingLayer

logger = logging.getLogger("vllm.lens")

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

"""CUDA-graph mode settings: read from the environment, stored in ``additional_config``, and
checked against each request before generation."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from vllm_lens._helpers._serialize import decode_layers

if TYPE_CHECKING:
    from vllm_lens._helpers.types import Hook

# The key of these settings in ``VllmConfig.additional_config``.
CONFIG_KEY = "vllm_lens"

# Stored with the settings, so part of vLLM's compile cache key. Increase it when a change
# alters what a compiled graph contains and no setting changes.
GRAPH_VERSION = 2


@dataclass(frozen=True)
class LensGraphConfig:
    """The CUDA-graph settings of one engine."""

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
        """Raise ``ValueError`` for a request that this CUDA-graph server cannot serve: the
        request's capture value, steering layers, hooks and Q/K flag."""
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
        layers = decode_layers(residual_stream)
        # "All layers" would return only the named ones, with no layer labels.
        if layers is None:
            raise ValueError(
                "Under VLLM_LENS_CUDAGRAPH, output_residual_stream must be a list of "
                f"layers from VLLM_LENS_CAPTURE_LAYERS: {list(self.capture_layers)}."
            )
        outside = sorted(set(layers) - set(self.capture_layers))
        if outside or not self.capture_layers:
            raise ValueError(
                f"output_residual_stream names layer(s) {outside}, but under "
                "VLLM_LENS_CUDAGRAPH this server captures only the layers in "
                f"VLLM_LENS_CAPTURE_LAYERS: {list(self.capture_layers)}."
            )

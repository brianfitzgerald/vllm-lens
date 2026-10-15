"""CUDA-graph mode settings: read from the environment, stored in ``additional_config``, and
checked against each request before generation."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from vllm_lens._helpers._serialize import decode_layers

if TYPE_CHECKING:
    from vllm_lens._helpers.types import Hook

# The key of these settings in ``VllmConfig.additional_config``, part of the compile cache key.
CONFIG_KEY = "vllm_lens"


@dataclass(frozen=True)
class LensGraphConfig:
    """The CUDA-graph settings of one engine: on when a layer list is set."""

    enabled: bool = False
    capture_layers: tuple[int, ...] = ()
    steer_layers: tuple[int, ...] = ()

    @classmethod
    def from_env(cls) -> LensGraphConfig:
        """Read the ``VLLM_LENS_CAPTURE_LAYERS`` and ``VLLM_LENS_STEER_LAYERS`` lists."""
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
            enabled=any(layers.values()),
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
        return {**others, CONFIG_KEY: asdict(self)}

    def reject_hooks(self, hooks: list[Hook], qk: bool = False) -> None:
        """Raise ``ValueError`` for hooks or Q/K capture, which need the forward hooks."""
        if self.enabled and (hooks or qk):
            raise ValueError(
                "The CUDA-graph mode (VLLM_LENS_*_LAYERS) is on, so the vllm-lens forward "
                "hooks do not run: hooks and Q/K capture are not available. Unset the "
                "layer lists to serve this request in eager mode."
            )

    def reject_request(self, residual_stream: Any, steering_layers: set[int]) -> None:
        """Raise ``ValueError`` for a capture or steering layer outside the armed lists, or
        for a capture of every layer."""
        if not self.enabled:
            return
        unsteered = sorted(steering_layers - set(self.steer_layers))
        if unsteered:
            raise ValueError(
                f"apply_steering_vectors names layer(s) {unsteered}, but this server "
                f"steers only the layers in VLLM_LENS_STEER_LAYERS: {list(self.steer_layers)}."
            )
        if residual_stream is None:
            return
        layers = decode_layers(residual_stream)
        # "All layers" would return only the named ones, with no layer labels.
        if layers is None:
            raise ValueError(
                "In the CUDA-graph mode, output_residual_stream must be a list of layers "
                f"from VLLM_LENS_CAPTURE_LAYERS: {list(self.capture_layers)}."
            )
        outside = sorted(set(layers) - set(self.capture_layers))
        if outside or not self.capture_layers:
            raise ValueError(
                f"output_residual_stream names layer(s) {outside}, but this server "
                "captures only the layers in VLLM_LENS_CAPTURE_LAYERS: "
                f"{list(self.capture_layers)}."
            )

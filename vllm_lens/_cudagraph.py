"""CUDA-graph mode settings, read from ``additional_config["vllm_lens"]`` and checked against
each request before generation."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm_lens._helpers.types import SteeringVector

# The key of these settings in ``VllmConfig.additional_config``, part of the compile cache key.
CONFIG_KEY = "vllm_lens"


@dataclass(frozen=True)
class LensGraphConfig:
    """The CUDA-graph settings of one engine: on when a layer list is set."""

    capture_layers: tuple[int, ...] = ()
    steer_layers: tuple[int, ...] = ()

    @property
    def enabled(self) -> bool:
        """Whether the engine runs in the CUDA-graph mode."""
        return bool(self.capture_layers or self.steer_layers)

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> LensGraphConfig:
        """Read ``additional_config["vllm_lens"]`` of an ``EngineArgs`` or ``VllmConfig``:
        ``{"capture_layers": [...], "steer_layers": [...]}``."""
        stored = (vllm_config.additional_config or {}).get(CONFIG_KEY)
        known = [field.name for field in fields(cls)]
        unknown = sorted(set(stored) - set(known)) if isinstance(stored, dict) else []
        if unknown:
            raise ValueError(
                f'additional_config["{CONFIG_KEY}"] has unknown key(s) {unknown}; '
                f"this vllm-lens version reads only {known}."
            )
        try:
            return cls(
                **{
                    key: tuple(sorted({int(layer) for layer in value}))
                    for key, value in (stored or {}).items()
                }
            )
        except (AttributeError, TypeError, ValueError):
            raise ValueError(
                f'additional_config["{CONFIG_KEY}"] must map capture_layers and '
                f"steer_layers to lists of layer numbers, got {stored!r}."
            ) from None

    def reject(
        self,
        hooks: object = None,
        qk: bool = False,
        residual_stream: Any = None,
        vectors: list[SteeringVector] | None = None,
    ) -> None:
        """Raise ``ValueError`` for what only the forward hooks serve: hooks, Q/K capture,
        a capture or steering layer outside the lists, or a capture of every layer."""
        if not self.enabled:
            return
        if hooks or qk:
            raise ValueError(
                "The CUDA-graph mode (additional_config['vllm_lens']) is on, so the vllm-lens forward "
                "hooks do not run: hooks and Q/K capture are not available. Remove the "
                "layer lists to serve this request in eager mode."
            )
        steering_layers = {layer for sv in vectors or [] for layer in sv.layer_indices}
        unsteered = sorted(steering_layers - set(self.steer_layers))
        if unsteered:
            raise ValueError(
                f"apply_steering_vectors names layer(s) {unsteered}, but this server "
                f"steers only its steer_layers: {list(self.steer_layers)}."
            )
        if residual_stream is None:
            return
        # "All layers" would return only the named ones, with no layer labels.
        if not isinstance(residual_stream, list):
            raise ValueError(
                "In the CUDA-graph mode, output_residual_stream must be a list of layers "
                f"from its capture_layers: {list(self.capture_layers)}."
            )
        outside = sorted(set(residual_stream) - set(self.capture_layers))
        if outside:
            raise ValueError(
                f"output_residual_stream names layer(s) {outside}, but this server "
                f"captures only its capture_layers: {list(self.capture_layers)}."
            )

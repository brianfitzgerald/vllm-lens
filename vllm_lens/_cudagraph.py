"""CUDA-graph mode settings, read from ``additional_config["vllm_lens"]`` and checked against
each request before generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm_lens._helpers._serialize import decode_layers

if TYPE_CHECKING:
    from vllm_lens._helpers.types import Hook, SteeringVector

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
        stored = (getattr(vllm_config, "additional_config", None) or {}).get(CONFIG_KEY)
        try:
            return cls(
                **{
                    key: tuple(sorted({int(layer) for layer in value}))
                    for key, value in (stored or {}).items()
                }
            )
        except (TypeError, ValueError):
            raise ValueError(
                f'additional_config["{CONFIG_KEY}"] must map capture_layers and '
                f"steer_layers to lists of layer numbers, got {stored!r}."
            ) from None

    def reject_hooks(self, hooks: list[Hook], qk: bool = False) -> None:
        """Raise ``ValueError`` for hooks or Q/K capture, which need the forward hooks."""
        if self.enabled and (hooks or qk):
            raise ValueError(
                "The CUDA-graph mode (additional_config['vllm_lens']) is on, so the vllm-lens forward "
                "hooks do not run: hooks and Q/K capture are not available. Remove the "
                "layer lists to serve this request in eager mode."
            )

    def reject_request(
        self, residual_stream: Any, vectors: list[SteeringVector] | None
    ) -> None:
        """Raise ``ValueError`` for a capture or steering layer outside the configured lists,
        or for a capture of every layer."""
        if not self.enabled:
            return
        steering_layers = {layer for sv in vectors or [] for layer in sv.layer_indices}
        unsteered = sorted(steering_layers - set(self.steer_layers))
        if unsteered:
            raise ValueError(
                f"apply_steering_vectors names layer(s) {unsteered}, but this server "
                f"steers only its steer_layers: {list(self.steer_layers)}."
            )
        if residual_stream is None:
            return
        layers = decode_layers(residual_stream)
        # "All layers" would return only the named ones, with no layer labels.
        if layers is None:
            raise ValueError(
                "In the CUDA-graph mode, output_residual_stream must be a list of layers "
                f"from its capture_layers: {list(self.capture_layers)}."
            )
        outside = sorted(set(layers) - set(self.capture_layers))
        if outside:
            raise ValueError(
                f"output_residual_stream names layer(s) {outside}, but this server "
                f"captures only its capture_layers: {list(self.capture_layers)}."
            )

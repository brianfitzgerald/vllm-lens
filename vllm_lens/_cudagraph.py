"""CUDA-graph mode settings, read from ``additional_config["vllm_lens"]`` and checked against
each request before generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator

if TYPE_CHECKING:
    from vllm_lens._helpers.types import SteeringRow, SteeringVector

# The key of these settings in ``VllmConfig.additional_config``, part of the compile cache key.
CONFIG_KEY = "vllm_lens"


class LensGraphConfig(BaseModel):
    """The CUDA-graph settings of one engine: on when a layer list is set."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    capture_layers: tuple[Annotated[int, Field(strict=True, ge=0)], ...] = ()
    steer_layers: tuple[Annotated[int, Field(strict=True, ge=0)], ...] = ()
    # False drops the norm buffer and its per-token norm: requests may not use norm_match.
    steer_norm_match: StrictBool = True

    @field_validator("capture_layers", "steer_layers")
    @classmethod
    def _sorted_layers(cls, layers: tuple[int, ...]) -> tuple[int, ...]:
        """Sort a layer list and remove its duplicates."""
        return tuple(sorted(set(layers)))

    @property
    def enabled(self) -> bool:
        """Whether the engine runs in the CUDA-graph mode."""
        return bool(self.capture_layers or self.steer_layers)

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> LensGraphConfig:
        """Read ``additional_config["vllm_lens"]`` of an ``EngineArgs`` or ``VllmConfig``."""
        return cls.model_validate(
            (vllm_config.additional_config or {}).get(CONFIG_KEY, {})
        )

    def reject(
        self,
        hooks: object = None,
        qk: bool = False,
        residual_stream: Any = None,
        vectors: list[SteeringVector] | None = None,
        rows: list[SteeringRow] | None = None,
    ) -> None:
        """Raise ``ValueError`` for what only the forward hooks serve: hooks, Q/K capture,
        a capture or steering layer outside the lists, a capture of every layer, or
        ``norm_match`` on a server without the norm buffer."""
        if not self.enabled:
            return
        if hooks or qk:
            raise ValueError(
                "The CUDA-graph mode (additional_config['vllm_lens']) is on, so the vllm-lens forward "
                "hooks do not run: hooks and Q/K capture are not available. Remove the "
                "layer lists to serve this request in eager mode."
            )
        for key, entries in (
            ("apply_steering_vectors", vectors),
            ("apply_steering_rows", rows),
        ):
            steering_layers = {
                layer for entry in entries or [] for layer in entry.layer_indices
            }
            unsteered = sorted(steering_layers - set(self.steer_layers))
            if unsteered:
                raise ValueError(
                    f"{key} names layer(s) {unsteered}, but this server "
                    f"steers only its steer_layers: {list(self.steer_layers)}."
                )
        if not self.steer_norm_match and any(sv.norm_match for sv in vectors or []):
            raise ValueError(
                "This server sets steer_norm_match to false, so a steering vector "
                "cannot use norm_match."
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

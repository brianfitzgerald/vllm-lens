"""Pydantic model of a vllm-lens steering vector."""

from __future__ import annotations

from typing import Any, Self

import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    field_serializer,
    field_validator,
    model_validator,
)

from vllm_lens._helpers._serialize import deserialize_tensor, serialize_tensor


class SteeringVector(BaseModel):
    """Steer with a vector added at chosen layers and positions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    activations: torch.Tensor
    """Steering activations of shape ``(n_layers, hidden_dim)``."""

    layer_indices: list[int]
    """Which model layers this steering vector applies to.  Length must
    match ``activations.shape[0]``."""

    scale: float = 1.0
    """Scalar multiplier applied to the steering vector before addition."""

    norm_match: bool = False
    """If True, scale the steering vector so the added magnitude equals the
    residual stream's per-token L2 norm (times ``scale``):
    ``h' = h + scale · ‖h‖ · v/‖v‖``.  Does NOT renormalize ``h'`` back to
    ``‖h‖`` — the residual norm grows accordingly."""

    position_indices: list[int] | None = None
    """Absolute token positions to steer; ``None`` steers every position."""

    @field_validator("activations", mode="before")
    @classmethod
    def _deserialize_activations(cls, v: Any) -> torch.Tensor:
        """Accept base64 dicts (from JSON transport) or raw tensors."""
        if isinstance(v, dict) and "data" in v:
            return deserialize_tensor(v)
        if isinstance(v, torch.Tensor):
            return v
        raise ValueError(
            f"activations must be a torch.Tensor or a base64 dict, got {type(v)}"
        )

    @field_serializer("activations")
    def _serialize_activations(self, v: torch.Tensor, _info: Any) -> dict[str, Any]:
        """Serialize tensor to base64 dict for JSON transport."""
        return serialize_tensor(v)

    @model_validator(mode="after")
    def _check_shape(self) -> Self:
        """Validate activation tensor shape matches layer_indices."""
        if self.activations.dim() != 2:
            raise ValueError(
                f"activations must be (n_layers, hidden_dim), got {self.activations.dim()}D"
            )
        if self.activations.shape[0] != len(self.layer_indices):
            raise ValueError(
                f"activations dim 0 ({self.activations.shape[0]}) must match "
                f"len(layer_indices) ({len(self.layer_indices)})"
            )
        return self

    @property
    def layer_index_map(self) -> dict[int, int]:
        """Maps actual model layer index to index into ``activations`` dim-0."""
        if not hasattr(self, "_layer_index_map_cache"):
            object.__setattr__(
                self,
                "_layer_index_map_cache",
                {li: i for i, li in enumerate(self.layer_indices)},
            )
        return self._layer_index_map_cache  # type: ignore[reportAttributeAccessIssue]

"""CUDA-graph mode: run the plugin without forcing ``enforce_eager``."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm_lens._helpers.types import Hook

# The key of these settings in ``VllmConfig.additional_config``.
CONFIG_KEY = "vllm_lens"

# In the stored settings, so in vLLM's compile cache key; nothing reads it. Increase
# it when a change alters what a compiled graph contains and no setting changes.
GRAPH_VERSION = 1


@dataclass(frozen=True)
class LensGraphConfig:
    """The CUDA-graph settings of one engine, read from the environment once.
    ``VllmConfig.additional_config`` holds them for the front end and the workers."""

    enabled: bool = False

    @classmethod
    def from_env(cls) -> LensGraphConfig:
        """Read ``VLLM_LENS_CUDAGRAPH``."""
        flag = os.environ.get("VLLM_LENS_CUDAGRAPH", "").strip().lower()
        return cls(enabled=flag in ("1", "true", "yes", "on"))

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> LensGraphConfig:
        """Read the settings that ``to_additional_config`` stored."""
        stored = (getattr(vllm_config, "additional_config", None) or {}).get(CONFIG_KEY)
        if not isinstance(stored, dict):
            return cls()
        # A user can write this key too, so a missing field takes its default.
        return cls(enabled=bool(stored.get("enabled", False)))

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
    ) -> None:
        """Raise ``ValueError`` for a request that this CUDA-graph server cannot serve.
        The arguments are the request's capture value, steering layers and hooks."""
        if self.enabled and (residual_stream is not None or steering_layers or hooks):
            raise ValueError(
                "VLLM_LENS_CUDAGRAPH is set, so the vllm-lens forward hooks do not "
                "run: activation capture, steering and hooks are not available. "
                "Unset VLLM_LENS_CUDAGRAPH to serve this request in eager mode."
            )

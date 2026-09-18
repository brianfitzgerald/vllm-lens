"""CUDA-graph mode: run the plugin without forcing ``enforce_eager``."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

# The key of these settings in ``VllmConfig.additional_config``.
CONFIG_KEY = "vllm_lens"


@dataclass(frozen=True)
class LensGraphConfig:
    """The CUDA-graph settings of one engine.

    Read from the environment when the engine config is created, and stored in
    ``VllmConfig.additional_config``. The front end and the workers of an engine
    then read the same values, and the values are part of the compile cache key.
    """

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

    def to_additional_config(self, existing: dict[str, Any] | None) -> dict[str, Any]:
        """``existing`` with these settings, or without them when the mode is off."""
        others = {
            key: value for key, value in (existing or {}).items() if key != CONFIG_KEY
        }
        return {**others, CONFIG_KEY: asdict(self)} if self.enabled else others

    def reject_unserved(self, needs_hooks: bool) -> None:
        """Raise for a request that needs forward hooks, which a CUDA graph skips."""
        if self.enabled and needs_hooks:
            raise RuntimeError(
                "VLLM_LENS_CUDAGRAPH is set, so the vllm-lens forward hooks do not "
                "run: activation capture, steering and hooks are not available. "
                "Unset VLLM_LENS_CUDAGRAPH to serve this request in eager mode."
            )

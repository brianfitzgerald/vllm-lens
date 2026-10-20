"""Position selection for residual-stream capture: ``output_residual_stream_positions``.

Positions index the model input: prompt tokens are ``0 .. n_prompt - 1``, and ``n_prompt + k``
is the state computed when generated token ``k`` is fed back.
"""

from __future__ import annotations

import json
from typing import Any

POSITIONS_KEY = "output_residual_stream_positions"
LAST_PROMPT_TOKEN = "last_prompt_token"
# The activations entry that lists the absolute position of each captured row.
POSITIONS_META_KEY = "residual_stream_positions"


def parse_capture_positions(value: Any) -> str | list[int] | None:
    """``"last_prompt_token"``, sorted unique non-negative positions, or None; a JSON string
    of either is accepted. Raise ``ValueError`` for anything else."""
    if value is None:
        return None
    if isinstance(value, str):
        if value == LAST_PROMPT_TOKEN:
            return LAST_PROMPT_TOKEN
        try:
            value = json.loads(value)
        except ValueError:
            raise ValueError(
                f"{POSITIONS_KEY} must be {LAST_PROMPT_TOKEN!r} or a list of "
                f"absolute token positions, got {value!r}"
            ) from None
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"{POSITIONS_KEY} must be {LAST_PROMPT_TOKEN!r} or a list of "
            f"absolute token positions, got {type(value).__name__}"
        )
    if not value:
        raise ValueError(f"{POSITIONS_KEY} must name at least one position")
    for position in value:
        if isinstance(position, bool) or not isinstance(position, int):
            raise ValueError(
                f"{POSITIONS_KEY} entries must be integers, got {position!r}"
            )
        if position < 0:
            raise ValueError(
                f"{POSITIONS_KEY} entries must be non-negative absolute positions, "
                f"got {position}"
            )
    return sorted(set(value))

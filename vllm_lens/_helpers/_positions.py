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
            decoded = json.loads(value)
        except ValueError:
            raise ValueError(
                f"{POSITIONS_KEY} must be {LAST_PROMPT_TOKEN!r} or a list of "
                f"absolute token positions, got {value!r}"
            ) from None
        if decoded is None or decoded == LAST_PROMPT_TOKEN:
            return decoded
        value = decoded
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


def normalize_capture_positions(extra_args: dict[str, Any]) -> None:
    """Replace ``extra_args[POSITIONS_KEY]`` with its parsed form, or remove it when None.
    Raise ``ValueError`` for a bad selector or one without ``output_residual_stream``."""
    if POSITIONS_KEY not in extra_args:
        return
    selector = parse_capture_positions(extra_args[POSITIONS_KEY])
    if selector is None:
        del extra_args[POSITIONS_KEY]
        return
    if extra_args.get("output_residual_stream") is None:
        raise ValueError(f"{POSITIONS_KEY} requires output_residual_stream")
    extra_args[POSITIONS_KEY] = selector


def positions_in_slice(
    selector: str | list[int], abs_start: int, n_tokens: int, n_prompt: int
) -> list[int]:
    """The selected absolute positions in ``[abs_start, abs_start + n_tokens)``."""
    wanted = [n_prompt - 1] if isinstance(selector, str) else selector
    return [p for p in wanted if abs_start <= p < abs_start + n_tokens]


def order_positions(positions: list[int]) -> tuple[list[int], list[int]]:
    """Ascending unique positions and the row to keep for each: a position computed twice
    (after a preemption) keeps its latest row."""
    latest = {position: row for row, position in enumerate(positions)}
    ordered = sorted(latest)
    return ordered, [latest[p] for p in ordered]

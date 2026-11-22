"""Position selection for residual-stream capture (issue #40).

By default the worker captures every processed token position. The optional
``output_residual_stream_positions`` entry in ``extra_args`` narrows this to
the last prompt token or to explicit absolute positions. The helpers here hold
the request-side validation; they import no vLLM so the plugin, the client and
the CPU unit tests can share them.

Positions index the tokenized model input. Prompt tokens occupy
``0 .. n_prompt - 1`` and ``n_prompt + k`` is the state computed when generated
token ``k`` is fed back for the next step. The final generated token is never
processed, so it has no row.
"""

from __future__ import annotations

import json
from typing import Any

POSITIONS_KEY = "output_residual_stream_positions"
"""``extra_args`` / ``vllm_xargs`` key carrying the selector."""

LAST_PROMPT_TOKEN = "last_prompt_token"
"""Selector value for the final token of the full prompt."""

POSITIONS_META_KEY = "residual_stream_positions"
"""Activations entry listing the absolute position of each captured row."""


def parse_capture_positions(value: Any) -> str | list[int] | None:
    """Normalise a selector value from any entry point.

    Accepts ``None`` (capture every position), the literal
    ``"last_prompt_token"``, a list of non-negative absolute positions, or the
    JSON-encoded string form of either (how ``vllm_xargs`` delivers values).
    Explicit positions come back sorted with duplicates removed. Anything else
    raises ``ValueError`` so a bad request fails before it is scheduled.
    """
    if value is None:
        return None
    if isinstance(value, str):
        if value == LAST_PROMPT_TOKEN:
            return LAST_PROMPT_TOKEN
        try:
            decoded = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            raise ValueError(
                f"{POSITIONS_KEY} must be {LAST_PROMPT_TOKEN!r} or a list of "
                f"absolute token positions, got {value!r}"
            ) from None
        if decoded is None:
            return None
        if decoded == LAST_PROMPT_TOKEN:
            return LAST_PROMPT_TOKEN
        value = decoded
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"{POSITIONS_KEY} must be {LAST_PROMPT_TOKEN!r} or a list of "
            f"absolute token positions, got {type(value).__name__}"
        )
    if not value:
        raise ValueError(f"{POSITIONS_KEY} must name at least one position")
    positions: set[int] = set()
    for position in value:
        if isinstance(position, bool) or not isinstance(position, int):
            raise ValueError(
                f"{POSITIONS_KEY} entries must be integers, got {position!r}"
            )
        if position < 0:
            raise ValueError(
                f"{POSITIONS_KEY} entries must be non-negative absolute "
                f"positions, got {position}"
            )
        positions.add(position)
    return sorted(positions)


def normalize_capture_positions(extra_args: dict[str, Any] | None) -> None:
    """Validate the selector at a request entry point and store its normal form.

    Rewrites ``extra_args[POSITIONS_KEY]`` in place so the workers receive the
    parsed value. A selector without ``output_residual_stream`` is rejected,
    since there would be nothing to select from.
    """
    if not extra_args or POSITIONS_KEY not in extra_args:
        return
    selector = parse_capture_positions(extra_args[POSITIONS_KEY])
    if selector is None:
        del extra_args[POSITIONS_KEY]
        return
    if extra_args.get("output_residual_stream") is None:
        raise ValueError(f"{POSITIONS_KEY} requires output_residual_stream")
    extra_args[POSITIONS_KEY] = selector

"""The position selector of residual-stream capture: parsing, slicing and ordering."""

import json

import pytest

from vllm_lens._helpers._positions import (
    LAST_PROMPT_TOKEN,
    POSITIONS_KEY,
    normalize_capture_positions,
    order_positions,
    parse_capture_positions,
    positions_in_slice,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("null", None),
        (LAST_PROMPT_TOKEN, LAST_PROMPT_TOKEN),
        (json.dumps(LAST_PROMPT_TOKEN), LAST_PROMPT_TOKEN),
        ([5, 2, 5], [2, 5]),
        ((5, 2), [2, 5]),
        ("[5, 2, 5]", [2, 5]),
        ([0], [0]),
    ],
)
def test_parse_capture_positions(value, expected) -> None:
    """Valid selectors parse to their normal form."""
    assert parse_capture_positions(value) == expected


@pytest.mark.parametrize(
    "value",
    [[-1], [1.5], [True], ["3"], [], "last", "{}", {"a": 1}, 3, "3"],
)
def test_parse_rejects_invalid_selectors(value) -> None:
    """Anything else raises ValueError."""
    with pytest.raises(ValueError, match=POSITIONS_KEY):
        parse_capture_positions(value)


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ({"output_residual_stream": [2], POSITIONS_KEY: "[5, 2, 5]"}, [2, 5]),
        ({"output_residual_stream": [2], POSITIONS_KEY: None}, None),
        ({"output_residual_stream": [2]}, None),
    ],
    ids=["list", "none", "absent"],
)
def test_normalize_rewrites_extra_args(extra: dict, expected) -> None:
    """The parsed selector replaces the value; None removes the key."""
    normalize_capture_positions(extra)
    assert extra.get(POSITIONS_KEY) == expected


def test_normalize_requires_residual_stream_capture() -> None:
    """A selector without output_residual_stream is rejected."""
    with pytest.raises(ValueError, match="requires output_residual_stream"):
        normalize_capture_positions({POSITIONS_KEY: LAST_PROMPT_TOKEN})


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        (LAST_PROMPT_TOKEN, [[], [], [9], [], []]),
        ([0, 3, 4, 9, 11, 500], [[0, 3], [4], [9], [], [11]]),
    ],
    ids=["last-prompt-token", "explicit"],
)
def test_positions_in_slice(selector, expected) -> None:
    """A 10-token prompt in chunks of 4, 4 and 2, then two decode steps."""
    slices = [(0, 4), (4, 4), (8, 2), (10, 1), (11, 1)]
    assert [positions_in_slice(selector, s, n, 10) for s, n in slices] == expected


def test_order_positions_keeps_the_latest_row() -> None:
    """A position computed twice keeps its latest row."""
    assert order_positions([3, 7, 3, 1]) == ([1, 3, 7], [3, 2, 1])
    assert order_positions([]) == ([], [])

"""The position selector of residual-stream capture: parsing and normalizing."""

import pytest

from vllm_lens._activations_plugin import _check_capture_options
from vllm_lens._helpers._positions import (
    LAST_PROMPT_TOKEN,
    POSITIONS_KEY,
    parse_capture_positions,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (LAST_PROMPT_TOKEN, LAST_PROMPT_TOKEN),
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
    [[-1], [1.5], [True], ["3"], [], "last", "{}", {"a": 1}, 3, "3", "null"],
)
def test_parse_rejects_invalid_selectors(value) -> None:
    """Anything else raises ValueError."""
    with pytest.raises(ValueError, match=POSITIONS_KEY):
        parse_capture_positions(value)


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        (
            {"output_residual_stream": "[2]", POSITIONS_KEY: "[5, 2, 5]"},
            {"output_residual_stream": [2], POSITIONS_KEY: [2, 5]},
        ),
        (
            {"output_residual_stream": [2], POSITIONS_KEY: None},
            {"output_residual_stream": [2]},
        ),
    ],
    ids=["strings", "none"],
)
def test_capture_options_are_normalized(extra: dict, expected: dict) -> None:
    """The layer list and the selector replace their string forms; None is removed."""
    _check_capture_options(extra, set())
    assert extra == expected

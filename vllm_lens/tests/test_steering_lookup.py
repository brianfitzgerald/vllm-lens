"""The per-step steering lookup needs no GPU: it maps request ids to stored vectors."""

from types import SimpleNamespace

import pytest

from vllm_lens._worker_ext import _find_steering_configs


def _extension(keys: list[str]) -> SimpleNamespace:
    """A stand-in extension whose stored vectors are the key names themselves."""
    return SimpleNamespace(_steering_data={key: [f"vector:{key}"] for key in keys})


@pytest.mark.parametrize(
    ("internal_id", "extra_args", "expected"),
    [
        ("0", {"_steering_id": "_steer_3"}, ["vector:_steer_3"]),
        ("0", {"_steering_id": "_steer_9"}, []),
        ("cmpl-abc-0", None, ["vector:cmpl-abc"]),
        ("cmpl-abc-0", {}, ["vector:cmpl-abc"]),
        ("cmpl-abc", None, ["vector:cmpl-abc"]),
        ("other-7", None, []),
    ],
    ids=[
        "offline",
        "offline-unknown",
        "async-suffix",
        "async-empty-extra",
        "exact",
        "miss",
    ],
)
def test_lookup(internal_id: str, extra_args: dict | None, expected: list[str]) -> None:
    """Offline ids resolve through the sentinel, async ids through the external id prefix."""
    extension = _extension(["_steer_3", "cmpl-abc"])
    assert _find_steering_configs(extension, internal_id, extra_args) == expected  # type: ignore[arg-type]


def test_lookup_with_nothing_stored() -> None:
    """An engine with no steering data answers at once."""
    assert _find_steering_configs(_extension([]), "cmpl-abc-0", None) == []  # type: ignore[arg-type]

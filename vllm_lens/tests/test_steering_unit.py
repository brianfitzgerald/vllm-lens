"""Steering and hook lookup by request id. No GPU."""

from types import SimpleNamespace

import pytest

from vllm_lens._worker_ext import _request_hooks, _steering_key


@pytest.mark.parametrize(
    ("internal_id", "extra_args", "expected"),
    [
        ("0", {"_steering_id": "_steer_3"}, "_steer_3"),
        ("0", {"_steering_id": "_steer_9"}, None),
        ("cmpl-abc-0", None, "cmpl-abc"),
        ("cmpl-abc-0", {}, "cmpl-abc"),
        ("cmpl-abc-x-0", None, None),
        ("other-7", None, None),
    ],
    ids=["offline", "offline-unknown", "async-suffix", "async-empty", "dashed", "miss"],
)
def test_lookup(internal_id: str, extra_args: dict | None, expected: str | None):
    """Offline ids resolve through the sentinel, async ids through the exact external id."""
    extension = SimpleNamespace(
        _steering_data={"_steer_3": [], "cmpl-abc": []},
        _hook_data={"_steer_3": ["offline"], "cmpl-abc": ["async"]},
    )
    assert _steering_key(extension, internal_id, extra_args) == expected  # type: ignore[arg-type]
    hook_args = extra_args and {"_hook_id": extra_args["_steering_id"]}
    hooks = _request_hooks(extension, internal_id, hook_args)  # type: ignore[arg-type]
    assert hooks == (extension._hook_data[expected] if expected else [])

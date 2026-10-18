"""CPU tests of the prefix-cache salt of steered and hooked requests."""

from types import SimpleNamespace

import pytest

import vllm_lens._activations_plugin as plugin


def test_steered_requests_share_a_salt_and_hooked_ones_do_not() -> None:
    """The same vector bytes give the same salt; each hooked request gets its own."""
    assert plugin._cache_salt(None, False) is None
    assert plugin._cache_salt(b"vectors", False) == plugin._cache_salt(
        b"vectors", False
    )
    assert plugin._cache_salt(b"vectors", False) != plugin._cache_salt(b"others", False)
    assert plugin._cache_salt(b"vectors", True) != plugin._cache_salt(b"vectors", True)


@pytest.mark.parametrize(
    ("extra", "existing", "salt"),
    [
        (None, None, None),
        ({}, "user", "user"),
        ({"_lens_salt": "lens-steer:a"}, None, "lens-steer:a"),
        ({"_lens_salt": "lens-steer:a"}, "user", "user|lens-steer:a"),
    ],
    ids=["no-extra", "unsalted", "salted", "joined"],
)
def test_process_inputs_joins_the_salt(
    monkeypatch: pytest.MonkeyPatch,
    extra: dict | None,
    existing: str | None,
    salt: str | None,
) -> None:
    """The salt in extra_args is joined onto the request's own cache_salt."""
    monkeypatch.setattr(
        plugin,
        "_original_process_inputs",
        lambda self, request_id, prompt, params: SimpleNamespace(cache_salt=existing),
    )
    params = SimpleNamespace(extra_args=extra)
    request = plugin._patched_process_inputs(None, "r", "p", params)
    assert request.cache_salt == salt

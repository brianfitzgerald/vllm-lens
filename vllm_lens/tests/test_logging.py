"""Tests for the ``vllm_lens`` logger handler that ``register()`` installs."""

import logging

import pytest

from vllm_lens._activations_plugin import _configure_logging


@pytest.fixture
def pkg_logger(monkeypatch):
    """The ``vllm_lens`` logger with no handler, restored after the test."""
    target = logging.getLogger("vllm_lens")
    monkeypatch.setattr(target, "handlers", [])
    monkeypatch.setattr(target, "propagate", True)
    level = target.level
    yield target
    target.setLevel(level)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, logging.INFO), ("debug", logging.DEBUG), ("VERBOSE", logging.INFO)],
    ids=["unset", "valid", "invalid"],
)
def test_configure_logging_attaches_one_handler(
    pkg_logger, monkeypatch, value, expected
):
    """Two calls leave one handler; an invalid level is ignored and does not raise."""
    monkeypatch.delenv("VLLM_LENS_LOG_LEVEL", raising=False)
    if value is not None:
        monkeypatch.setenv("VLLM_LENS_LOG_LEVEL", value)

    _configure_logging()
    _configure_logging()

    assert len(pkg_logger.handlers) == 1
    assert pkg_logger.level == expected
    assert pkg_logger.propagate is False


def test_configure_logging_keeps_an_existing_handler(pkg_logger, monkeypatch):
    """A handler the application attached first is kept, and the level still applies."""
    existing = logging.NullHandler()
    pkg_logger.addHandler(existing)
    monkeypatch.setenv("VLLM_LENS_LOG_LEVEL", "WARNING")

    _configure_logging()

    assert pkg_logger.handlers == [existing]
    assert pkg_logger.propagate is True
    assert pkg_logger.level == logging.WARNING

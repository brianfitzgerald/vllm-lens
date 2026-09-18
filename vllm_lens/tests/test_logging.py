"""Tests for the ``vllm_lens`` logger handler that ``register()`` installs."""

import logging

import pytest

from vllm_lens._activations_plugin import _configure_logging


@pytest.mark.parametrize("level", ["INFO", "DEBUG"])
def test_configure_logging_attaches_one_handler(monkeypatch, level):
    """Two calls leave one handler, at the level ``VLLM_LENS_LOG_LEVEL`` names."""
    pkg_logger = logging.getLogger("vllm_lens")
    monkeypatch.setattr(pkg_logger, "handlers", [])
    monkeypatch.setattr(pkg_logger, "level", pkg_logger.level)
    monkeypatch.setenv("VLLM_LENS_LOG_LEVEL", level)

    _configure_logging()
    _configure_logging()

    assert len(pkg_logger.handlers) == 1
    assert pkg_logger.level == getattr(logging, level)


def test_configure_logging_keeps_an_existing_handler(monkeypatch):
    """A handler the application attached first is left alone."""
    pkg_logger = logging.getLogger("vllm_lens")
    existing = logging.NullHandler()
    monkeypatch.setattr(pkg_logger, "handlers", [existing])

    _configure_logging()

    assert pkg_logger.handlers == [existing]

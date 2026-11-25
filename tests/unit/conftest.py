"""Fixtures for the plugin tests without vLLM or CUDA."""

from types import SimpleNamespace
from typing import Any

import pytest

# Layers 0 to 3 armed for capture and steering.
ARMED = SimpleNamespace(
    additional_config={
        "vllm_lens": {"capture_layers": [0, 1, 2, 3], "steer_layers": [0, 1, 2, 3]}
    },
    model_config=SimpleNamespace(hf_text_config=SimpleNamespace(num_hidden_layers=4)),
)


def _engine(collective_rpc: Any, *, offline: bool, **attrs: Any) -> SimpleNamespace:
    """An offline ``LLM`` or an ``AsyncLLM`` stand-in on the armed config."""
    config = attrs.pop("vllm_config", ARMED)
    where = (
        {"llm_engine": SimpleNamespace(vllm_config=config)}
        if offline
        else {"vllm_config": config}
    )
    return SimpleNamespace(
        collective_rpc=collective_rpc,
        **where,
        **attrs,
    )


@pytest.fixture
def stub_engine():
    """The factory of engine stand-ins."""
    return _engine

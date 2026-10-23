"""Fixtures for worker tests without vLLM or CUDA."""

import importlib
import sys
from types import ModuleType

import pytest


@pytest.fixture(scope="module")
def worker():
    """Load the worker with the required vLLM imports."""
    with pytest.MonkeyPatch.context() as patch:
        loaded = set(sys.modules)
        context = ModuleType("vllm.forward_context")
        context.get_forward_context = lambda: None
        context.is_forward_context_available = lambda: False
        models = ModuleType("vllm.model_executor.models.utils")
        models.PPMissingLayer = type("PPMissingLayer", (), {})
        patch.setitem(sys.modules, context.__name__, context)
        patch.setitem(sys.modules, models.__name__, models)
        yield importlib.import_module("vllm_lens._worker_ext")
        for name in set(sys.modules) - loaded:
            if name.startswith("vllm_lens._worker_ext"):
                del sys.modules[name]

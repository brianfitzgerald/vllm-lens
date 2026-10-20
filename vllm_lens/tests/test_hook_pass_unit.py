"""The per-pass hook plan and the early exit of an idle layer. No GPU."""

from types import SimpleNamespace

import pytest
import torch

import vllm_lens._worker_ext as worker_ext
from vllm_lens._worker_ext import _hook_inner, _hook_pass, _HookPass, _pre_hook_inner

from .conftest import fake_extension


@pytest.mark.parametrize(
    ("extras", "hook_data", "persistent", "expected"),
    [
        ({}, {}, [], _HookPass(frozenset(), False)),
        ({"r-0": {"output_residual_stream": True}}, {}, [], _HookPass(None, False)),
        (
            {
                "r-0": {"output_residual_stream": [3]},
                "r-1": {"output_residual_stream": [5]},
            },
            {},
            [],
            _HookPass(frozenset({3, 5}), False),
        ),
        ({"r-0": {"_hook_id": "h"}}, {"h": ["hook"]}, [], _HookPass(frozenset(), True)),
        ({"r-0": None}, {}, ["hook"], _HookPass(frozenset(), True)),
    ],
    ids=["empty", "all-layers", "layer-lists", "request-hook", "persistent-hook"],
)
def test_hook_pass(extras, hook_data, persistent, expected) -> None:
    """One scan of the batch gives the captured layers and whether any hook runs."""
    extension = fake_extension(extras)
    extension._hook_data = hook_data
    extension._persistent_hooks = persistent
    assert _hook_pass(extension) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize("layer", [2, 3], ids=["idle", "captured"])
def test_an_idle_layer_is_skipped(monkeypatch: pytest.MonkeyPatch, layer: int) -> None:
    """A layer that no request captures, steers or hooks returns before the batch layout."""
    reached: list[int] = []

    def layout(runner, warn):
        reached.append(1)
        return None

    monkeypatch.setattr(worker_ext, "_batch_layout", layout)
    extension = SimpleNamespace(
        model_runner=None,
        _steer_pass=None,
        _hook_pass=_HookPass(frozenset({3}), False),
        _should_capture=True,
    )
    assert _hook_inner(extension, layer, torch.zeros(1, 8)) is None  # type: ignore[arg-type]
    assert _pre_hook_inner(extension, layer, torch.zeros(1, 8)) is None  # type: ignore[arg-type]
    assert len(reached) == (1 if layer == 3 else 0)

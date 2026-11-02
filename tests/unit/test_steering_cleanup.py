"""Prepared steering is cleared when generation fails or is cancelled."""

from types import SimpleNamespace

import pytest

import vllm_lens._activations_plugin as plugin


@pytest.mark.parametrize("method", ["generate", "chat"])
def test_offline_rows_clear_on_error(monkeypatch, method):
    """Both offline entry points clear table rows when generation raises."""
    calls = []
    config = SimpleNamespace(
        additional_config=None,
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(num_hidden_layers=4)
        ),
    )
    engine = SimpleNamespace(
        _hooks_installed=True,
        _lens_steering_tables={"t": 1},
        llm_engine=SimpleNamespace(vllm_config=config),
        collective_rpc=lambda name, args=(): calls.append((name, args)),
    )
    params = SimpleNamespace(
        n=2,
        skip_reading_prefix_cache=False,
        extra_args={
            "apply_steering_rows": [{"table": "t", "row": 0, "layer_indices": [1]}]
        },
    )

    def fail(*args, **kwargs):
        raise RuntimeError("generation failed")

    monkeypatch.setattr(plugin, f"_original_llm_{method}", fail)
    with pytest.raises(RuntimeError, match="generation failed"):
        getattr(plugin, f"_patched_llm_{method}")(engine, ["prompt"], params)
    assert ("clear_steering_data", ("_steer_0",)) in calls


async def test_async_rows_clear_on_close(monkeypatch):
    """Closing an active async request releases its prepared rows."""
    import sys
    from types import ModuleType

    module = ModuleType("vllm.v1.engine")
    module.EngineCoreRequest = type("EngineCoreRequest", (), {})
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", module)
    calls = []

    async def rpc(name, args=()):
        calls.append((name, args))
        return []

    async def original(*args, **kwargs):
        yield SimpleNamespace(finished=False)

    monkeypatch.setattr(plugin, "_original_generate", original)
    config = SimpleNamespace(
        additional_config=None,
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(num_hidden_layers=4)
        ),
    )
    engine = SimpleNamespace(
        _hooks_installed=True,
        _lens_steering_tables={"t": 1},
        vllm_config=config,
        collective_rpc=rpc,
    )
    params = SimpleNamespace(
        n=2,
        skip_reading_prefix_cache=False,
        extra_args={
            "apply_steering_rows": [{"table": "t", "row": 0, "layer_indices": [1]}]
        },
    )
    stream = plugin._patched_generate(engine, "prompt", params, "r")
    await anext(stream)
    await stream.aclose()
    assert ("clear_steering_data", ("r",)) in calls

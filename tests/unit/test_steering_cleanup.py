"""Table versions are released when generation fails or is cancelled."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

import vllm_lens._activations_plugin as plugin


def _params() -> SimpleNamespace:
    """Parameters that steer with row 0 of table ``t``, in two parallel samples."""
    return SimpleNamespace(
        n=2,
        skip_reading_prefix_cache=False,
        extra_args={
            "apply_steering_vectors": [{"table": "t", "row": 0, "layer_indices": [1]}]
        },
    )


@pytest.mark.parametrize("method", ["generate", "chat"])
def test_offline_rows_release_on_error(monkeypatch, stub_engine, method):
    """Both offline entry points send rows with no RPC and release them on error."""
    calls = []
    engine = stub_engine(lambda name, args=(): calls.append((name, args)), offline=True)
    engine._lens_steering_tables = {"t": (1, 0)}
    params = _params()
    sent = []

    def fail(self, prompts, params, *args, **kwargs):
        sent.append(params)
        raise RuntimeError("generation failed")

    monkeypatch.setattr(plugin, f"_original_llm_{method}", fail)
    with pytest.raises(RuntimeError, match="generation failed"):
        getattr(plugin, f"_patched_llm_{method}")(engine, ["prompt"], params)
    assert sent[0].extra_args["_steer_rows"] == [["t", 0, 0, 1.0, [1], False]]
    assert not calls and not engine._lens_table_users


@pytest.mark.parametrize("replaced", [False, True])
async def test_async_rows_release_on_close(monkeypatch, stub_engine, replaced):
    """Closing an active request releases its table version; a version replaced while
    the request ran is then dropped on the workers."""
    module = ModuleType("vllm.v1.engine")
    module.EngineCoreRequest = type("EngineCoreRequest", (), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", module)
    calls = []

    async def rpc(name, args=()):
        calls.append((name, args))
        return []

    async def original(*args, **kwargs):
        yield SimpleNamespace(finished=False)

    monkeypatch.setattr(plugin, "_original_generate", original)
    engine = stub_engine(rpc, offline=False)
    engine._lens_steering_tables = {"t": (1, 0)}
    stream = plugin._patched_generate(engine, "prompt", _params(), "r")
    await anext(stream)
    assert engine._lens_table_users[("t", 0)] == 1
    if replaced:
        assert plugin._replace_table(engine, "t", 1) == (1, None)
    await stream.aclose()  # type: ignore[attr-defined]
    assert not engine._lens_table_users
    assert calls == ([("clear_steering_table", ("t", 0))] if replaced else [])

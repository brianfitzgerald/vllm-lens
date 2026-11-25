"""CPU tests of the prefix-cache rule for intervention requests."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import vllm_lens._activations_plugin as plugin
from vllm_lens import SteeringVector

INLINE = {
    "apply_steering_vectors": [
        SteeringVector(activations=torch.ones(1, 4), layer_indices=[0])
    ]
}
# Each kind's extra_args, whether it skips the cache read and whether it gets a salt.
KINDS = {
    "ordinary": ({}, False, False),
    "inline-steering": (INLINE, False, True),
    "table-steering": (
        {"apply_steering_vectors": [{"table": "t", "row": 1, "layer_indices": [0]}]},
        False,
        True,
    ),
    "last-token-capture": (
        {"output_residual_stream": [0], plugin.POSITIONS_KEY: "last_prompt_token"},
        False,
        False,
    ),
    "full-capture": ({"output_residual_stream": [0]}, True, False),
    "list-capture": (
        {"output_residual_stream": [0], plugin.POSITIONS_KEY: [0, 2]},
        True,
        False,
    ),
    "mean-capture": (
        {"output_residual_stream": [0], "output_residual_stream_pool": "mean"},
        True,
        False,
    ),
    "steered-last-token-capture": (
        {
            **INLINE,
            "output_residual_stream": [0],
            plugin.POSITIONS_KEY: "last_prompt_token",
        },
        False,
        True,
    ),
    "explicit-skip": ({**INLINE, "skip_reading_prefix_cache": True}, True, True),
}


@pytest.mark.parametrize("existing", [None, "user"])
@pytest.mark.parametrize("salt", [None, "lens-steer:abc"])
def test_process_inputs_adds_the_steering_salt(monkeypatch, existing, salt):
    """A steered request's salt follows the user salt; other requests keep theirs."""
    monkeypatch.setattr(
        plugin,
        "_original_process_inputs",
        lambda *a, **k: SimpleNamespace(cache_salt=existing),
    )
    params = SimpleNamespace(extra_args={"_lens_cache_salt": salt})
    request = plugin._patched_process_inputs(None, "0", "p", params)
    if salt is None:
        assert request.cache_salt == existing
    else:
        assert request.cache_salt == (f"{existing}|{salt}" if existing else salt)


@pytest.mark.parametrize("built", [False, True])
@pytest.mark.parametrize("kind", KINDS)
async def test_async_intervention_params(
    monkeypatch, tmp_path, stub_engine, built, kind
):
    """Each kind skips the read and gets a salt as KINDS says, on a copy of the caller's
    parameters."""
    monkeypatch.setenv(plugin.CAPTURE_DIR_ENV, str(tmp_path))
    module = ModuleType("vllm.v1.engine")
    request_type = type("EngineCoreRequest", (), {})
    module.EngineCoreRequest = request_type
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", module)
    extra, skips, salted = KINDS[kind]
    params = SimpleNamespace(extra_args=dict(extra), skip_reading_prefix_cache=False)
    prompt = request_type() if built else "p"
    if built:
        prompt.sampling_params = SimpleNamespace(
            extra_args=dict(extra), skip_reading_prefix_cache=False
        )
        prompt.cache_salt = "user"
    received = []

    async def original(self, prompt, sent, request_id, **kwargs):
        received.append(prompt.sampling_params if built else sent)
        yield SimpleNamespace(finished=False)

    async def rpc(*args, **kwargs):
        return []

    monkeypatch.setattr(plugin, "_original_generate", original)
    engine = stub_engine(rpc, offline=False)
    engine._lens_steering_tables = {"t": (2, 5)}
    async for _ in plugin._patched_generate(engine, prompt, params, "r"):
        pass
    sent = received[0]
    salt = sent.extra_args.get("_lens_cache_salt")
    assert (salt is not None) == salted
    assert sent.skip_reading_prefix_cache == skips
    assert params.extra_args == extra
    assert not params.skip_reading_prefix_cache
    if built:
        assert prompt.cache_salt == (f"user|{salt}" if salted else "user")


def test_the_salt_names_the_steering(stub_engine):
    """The same steering gets the same salt; another scale, row, layer, position, vector or
    table version gets another."""
    engine = stub_engine(None, offline=True)

    def salt(vectors: list[dict], version: int = 5) -> str:
        engine._lens_steering_tables = {"t": (4, version)}
        params = SimpleNamespace(skip_reading_prefix_cache=False)
        extra = {"apply_steering_vectors": vectors}
        plugin._prepare_request(
            engine, engine.llm_engine.vllm_config, params, extra, "k"
        )
        assert not params.skip_reading_prefix_cache
        return extra["_lens_cache_salt"]

    row = {"table": "t", "row": 1, "layer_indices": [0], "scale": 2.0}
    inline = {"activations": torch.ones(1, 4), "layer_indices": [0]}
    salts = [
        salt([row]),
        salt([{**row, "scale": 3.0}]),
        salt([{**row, "row": 2}]),
        salt([{**row, "layer_indices": [1]}]),
        salt([row], version=6),
        salt([inline]),
        salt([{**inline, "activations": torch.full((1, 4), 2.0)}]),
        salt([{**inline, "position_indices": [3]}]),
        salt([row, inline]),
    ]
    assert salt([row]) == salts[0] and salt([inline]) == salts[5]
    assert len(set(salts)) == len(salts)


def test_a_table_registered_again_gets_a_new_version(stub_engine):
    """A version names one set of contents, even after the table is removed."""
    engine = stub_engine(None, offline=True)
    first, _ = plugin._replace_table(engine, "t", 2)
    plugin._remove_table(engine, "t")
    second, _ = plugin._replace_table(engine, "t", 2)
    assert first != second


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("many", [False, True])
def test_offline_intervention_params(stub_engine, kind, many):
    """Copied offline parameters follow the same rule."""
    extra, skips, salted = KINDS[kind]
    params = SimpleNamespace(extra_args=dict(extra), skip_reading_prefix_cache=False)
    engine = stub_engine(lambda *a, **k: [], offline=True)
    engine._lens_steering_tables = {"t": (2, 5)}
    state = plugin._prepare_offline_params(engine, [params, params] if many else params)
    sent = state["params"] if many else [state["params"]]
    for item in sent:
        assert item is not params
        assert ("_lens_cache_salt" in item.extra_args) == salted
        assert item.skip_reading_prefix_cache == skips
    assert params.extra_args == extra
    assert not params.skip_reading_prefix_cache

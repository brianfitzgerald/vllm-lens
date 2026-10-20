"""CPU tests of prefix-cache isolation for intervention requests."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import vllm_lens._activations_plugin as plugin
from vllm_lens import Hook, SteeringVector


@pytest.mark.parametrize("existing", [None, "user"])
@pytest.mark.parametrize("isolated", [False, True])
def test_process_inputs_isolates_each_request(monkeypatch, existing, isolated):
    """Assert that each intervention gets a unique salt and keeps the user salt."""
    monkeypatch.setattr(
        plugin,
        "_original_process_inputs",
        lambda *a, **k: SimpleNamespace(cache_salt=existing),
    )
    params = SimpleNamespace(extra_args={"_lens_isolate_cache": isolated})
    requests = [
        plugin._patched_process_inputs(None, str(i), "p", params) for i in range(2)
    ]
    if isolated:
        prefix = f"{existing}|lens-request:" if existing else "lens-request:"
        assert all(request.cache_salt.startswith(prefix) for request in requests)
        assert requests[0].cache_salt != requests[1].cache_salt
    else:
        assert all(request.cache_salt == existing for request in requests)
    assert params.extra_args == {"_lens_isolate_cache": isolated}


@pytest.mark.parametrize("built", [False, True])
@pytest.mark.parametrize(
    "kind", ["ordinary", "steering", "hooks", "persistent", "capture"]
)
async def test_async_intervention_params(monkeypatch, built, kind):
    """Assert isolation and cache bypass without changing caller parameters."""
    module = ModuleType("vllm.v1.engine")
    request_type = type("EngineCoreRequest", (), {})
    module.EngineCoreRequest = request_type
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", module)
    extras = {
        "ordinary": {},
        "steering": {
            "apply_steering_vectors": [
                SteeringVector(activations=torch.ones(1, 4), layer_indices=[0])
            ]
        },
        "hooks": {"apply_hooks": [Hook(fn=torch.zeros_like, layer_indices=[0])]},
        "persistent": {},
        "capture": {"output_residual_stream": [0]},
    }
    params = SimpleNamespace(extra_args=extras[kind], skip_reading_prefix_cache=False)
    original_extra = dict(params.extra_args)
    prompt = request_type() if built else "p"
    if built:
        prompt.sampling_params = SimpleNamespace(
            extra_args=dict(original_extra), skip_reading_prefix_cache=False
        )
        prompt.cache_salt = "user"
    received = []

    async def original(self, prompt, sent, request_id, **kwargs):
        received.append(prompt.sampling_params if built else sent)
        yield SimpleNamespace(finished=False)

    async def rpc(*args, **kwargs):
        return []

    monkeypatch.setattr(plugin, "_original_generate", original)
    engine = SimpleNamespace(
        collective_rpc=rpc,
        _lens_projections=frozenset(),
        _hooks_installed=True,
        _has_persistent_hooks=kind == "persistent",
    )
    async for _ in plugin._patched_generate(engine, prompt, params, "r"):
        pass
    sent = received[0]
    isolated = kind in ("steering", "hooks", "persistent")
    assert bool(sent.extra_args.get("_lens_isolate_cache")) == isolated
    assert sent.skip_reading_prefix_cache == (kind != "ordinary")
    assert params.extra_args == original_extra
    assert not params.skip_reading_prefix_cache
    if built:
        assert (
            prompt.cache_salt.startswith("user|lens-request:")
            if isolated
            else prompt.cache_salt == "user"
        )


@pytest.mark.parametrize(
    "kind", ["ordinary", "steering", "hooks", "persistent", "capture"]
)
@pytest.mark.parametrize("many", [False, True])
def test_offline_intervention_params(kind, many):
    """Assert that copied offline parameters carry the isolation flag."""
    extras = {
        "ordinary": {},
        "steering": {
            "apply_steering_vectors": [
                SteeringVector(activations=torch.ones(1, 4), layer_indices=[0])
            ]
        },
        "hooks": {"apply_hooks": [Hook(fn=torch.zeros_like, layer_indices=[0])]},
        "persistent": {},
        "capture": {"output_residual_stream": [0]},
    }
    params = SimpleNamespace(extra_args=extras[kind], skip_reading_prefix_cache=False)
    original_extra = dict(params.extra_args)
    engine = SimpleNamespace(
        collective_rpc=lambda *a, **k: [],
        _lens_projections=frozenset(),
        _hooks_installed=True,
        _has_persistent_hooks=kind == "persistent",
    )
    state = plugin._prepare_offline_params(engine, [params, params] if many else params)
    sent = state["params"] if many else [state["params"]]
    for item in sent:
        assert item is not params
        assert bool(item.extra_args.get("_lens_isolate_cache")) == (
            kind in ("steering", "hooks", "persistent")
        )
        assert item.skip_reading_prefix_cache == (kind != "ordinary")
    assert params.extra_args == original_extra
    assert not params.skip_reading_prefix_cache


def test_offline_persistent_hooks_use_model_defaults():
    """Assert that omitted parameters retain model defaults and get isolation."""
    defaults = SimpleNamespace(
        extra_args=None, skip_reading_prefix_cache=False, temperature=0.7
    )
    engine = SimpleNamespace(
        _has_persistent_hooks=True,
        _hooks_installed=True,
        get_default_sampling_params=lambda: defaults,
        _lens_projections=frozenset(),
    )
    sent = plugin._prepare_offline_params(engine, None)["params"]
    assert sent is not defaults and sent.temperature == 0.7
    assert sent.extra_args == {"_lens_isolate_cache": True}
    assert sent.skip_reading_prefix_cache
    assert defaults.extra_args is None and not defaults.skip_reading_prefix_cache

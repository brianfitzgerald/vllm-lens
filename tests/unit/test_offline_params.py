"""CPU tests for the plugin entry points: payloads ride in a copy of extra_args."""

import pickle
from types import SimpleNamespace

import cloudpickle
import pytest
import torch

import vllm_lens._activations_plugin as plugin
from vllm_lens import Hook, SteeringVector
from vllm_lens._helpers.types import HOOKS_PAYLOAD, STEERING_PAYLOAD


def _processor(persistent: bool = False) -> SimpleNamespace:
    """An InputProcessor with a 4-layer model."""
    return SimpleNamespace(
        model_config=SimpleNamespace(get_total_num_hidden_layers=lambda: 4),
        _lens_persistent_hooks=persistent,
    )


@pytest.fixture
def received(monkeypatch: pytest.MonkeyPatch) -> list:
    """The params the original process_inputs gets."""
    seen: list = []
    monkeypatch.setattr(
        plugin,
        "_original_process_inputs",
        lambda self, request_id, prompt, params, *a, **k: (
            seen.append(params) or SimpleNamespace(cache_salt=None)
        ),
    )
    return seen


def test_process_inputs_moves_payloads_into_a_copy(received: list) -> None:
    """The caller's params are unchanged; the copy carries pickled vectors and hooks."""
    storage = torch.ones(8, 4)
    vector = SteeringVector(activations=storage[:1], layer_indices=[0])
    hook = Hook(fn=lambda ctx, h: None, layer_indices=[3])
    extra = {
        "apply_steering_vectors": [vector],
        "apply_hooks": [hook],
        "activations_transport": "binary",
        "keep": 1,
    }
    params = SimpleNamespace(extra_args=dict(extra), skip_reading_prefix_cache=False)

    plugin._patched_process_inputs(_processor(), "r", "p", params)

    assert params.extra_args == extra and params.skip_reading_prefix_cache is False
    (sent,) = received
    assert set(sent.extra_args) == {"keep", STEERING_PAYLOAD, HOOKS_PAYLOAD}
    assert sent.skip_reading_prefix_cache is True
    (decoded,) = pickle.loads(sent.extra_args[STEERING_PAYLOAD])
    assert decoded.activations.untyped_storage().nbytes() == 4 * 4
    assert cloudpickle.loads(sent.extra_args[HOOKS_PAYLOAD])[0].layer_indices == [3]


def test_process_inputs_drops_client_payloads(received: list) -> None:
    """A client cannot send its own pickled payloads to the workers."""
    params = SimpleNamespace(
        extra_args={STEERING_PAYLOAD: b"x", HOOKS_PAYLOAD: b"y", "keep": 1},
        skip_reading_prefix_cache=False,
    )
    plugin._patched_process_inputs(_processor(), "r", "p", params)
    assert received[0].extra_args == {"keep": 1}


@pytest.mark.parametrize(
    ("extra", "persistent", "skip"),
    [
        ({}, False, False),
        ({"keep": 1}, False, False),
        ({"output_residual_stream": [0]}, False, True),
        ({"output_qk": True}, False, True),
        ({"skip_reading_prefix_cache": True}, False, True),
        ({}, True, True),
    ],
    ids=["none", "plain", "capture", "qk", "explicit", "persistent"],
)
def test_process_inputs_skips_the_prefix_cache(
    received: list, extra: dict, persistent: bool, skip: bool
) -> None:
    """Every request the hooks must see skips the prefix cache."""
    params = SimpleNamespace(extra_args=extra, skip_reading_prefix_cache=False)
    plugin._patched_process_inputs(_processor(persistent), "r", "p", params)
    assert received[0].skip_reading_prefix_cache is skip
    assert "skip_reading_prefix_cache" not in (received[0].extra_args or {})


@pytest.mark.parametrize(
    ("extra", "persistent", "salt"),
    [
        ({}, False, None),
        ({"output_residual_stream": [0]}, False, None),
        ({"apply_steering_vectors": "same"}, False, "lens-steer:"),
        ({"apply_hooks": True}, False, "lens-request:r"),
        ({}, True, "lens-request:r"),
    ],
    ids=["none", "capture", "steer", "hooks", "persistent"],
)
def test_process_inputs_salts_changed_kv(
    monkeypatch: pytest.MonkeyPatch, extra: dict, persistent: bool, salt: str | None
) -> None:
    """Steered or hooked requests write KV blocks only other requests like them can read."""
    requests: list = []

    def original(self, request_id, prompt, params, *args, **kwargs):
        requests.append(SimpleNamespace(cache_salt="user"))
        return requests[-1]

    monkeypatch.setattr(plugin, "_original_process_inputs", original)
    if extra.get("apply_steering_vectors"):
        extra = {
            "apply_steering_vectors": [
                SteeringVector(activations=torch.ones(1, 4), layer_indices=[0])
            ]
        }
    if extra.get("apply_hooks"):
        extra = {"apply_hooks": [Hook(fn=lambda ctx, h: None, layer_indices=[0])]}
    for _ in range(2):
        params = SimpleNamespace(
            extra_args=dict(extra), skip_reading_prefix_cache=False
        )
        plugin._patched_process_inputs(_processor(persistent), "r", "p", params)
    first, second = (request.cache_salt for request in requests)
    if salt is None:
        assert first == "user"
    else:
        assert first.startswith(f"user|{salt}") and first == second


def test_process_inputs_rejects_a_layer_out_of_range(received: list) -> None:
    """A layer the model does not have fails before the request reaches the engine."""
    vector = SteeringVector(activations=torch.ones(1, 4), layer_indices=[4])
    params = SimpleNamespace(
        extra_args={"apply_steering_vectors": [vector]}, skip_reading_prefix_cache=False
    )
    with pytest.raises(ValueError, match="out of range"):
        plugin._patched_process_inputs(_processor(), "r", "p", params)
    assert received == []


class _FakeLLM:
    """Records the RPCs."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._hooks_installed = True
        self.llm_engine = SimpleNamespace(input_processor=_processor())

    def collective_rpc(self, method: str, args: tuple = ()) -> list:
        """Record the method name."""
        self.calls.append(method)
        return []


@pytest.mark.parametrize("fail", [False, True], ids=["ok", "engine-failed"])
def test_offline_clears_worker_state_only_after_a_failure(
    monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    """A failed call drops every capture; a finished one collects and clears nothing."""

    def original(self, prompts, sampling_params, *args, **kwargs):
        if fail:
            raise RuntimeError("engine failed")
        return []

    monkeypatch.setattr(plugin, "_original_llm_generate", original)
    llm = _FakeLLM()
    params = SimpleNamespace(extra_args={"output_residual_stream": [0]})
    if fail:
        with pytest.raises(RuntimeError, match="failed"):
            plugin._patched_llm_generate(llm, ["p"], params)  # type: ignore[arg-type]
        assert llm.calls == ["clear_request_state"]
    else:
        plugin._patched_llm_generate(llm, ["p"], params)  # type: ignore[arg-type]
        assert llm.calls == ["get_captured_states_batch"]


class _FakeAsyncEngine:
    """Records async worker RPCs."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._hooks_installed = True
        self.input_processor = _processor()

    async def collective_rpc(self, method: str, args: tuple = ()) -> list:
        """Record the method name."""
        self.calls.append(method)
        return []


@pytest.mark.parametrize(
    ("extra", "calls"),
    [
        ({}, []),
        ({"apply_steering_vectors": "[]"}, []),
        (
            {"output_residual_stream": [0], "apply_hooks": "[]"},
            ["get_captured_states", "get_hook_results", "clear_request_state"],
        ),
    ],
    ids=["plain", "steer", "capture-hooks"],
)
async def test_async_generate_collects_and_clears(
    monkeypatch: pytest.MonkeyPatch, extra: dict, calls: list[str]
) -> None:
    """Steering needs no RPC; a capture or a hook is collected, then cleared."""

    async def original(self, prompt, sampling_params, request_id, **kwargs):
        yield SimpleNamespace(
            finished=True, prompt_token_ids=[1], outputs=[SimpleNamespace(token_ids=[])]
        )

    monkeypatch.setattr(plugin, "_original_generate", original)
    engine = _FakeAsyncEngine()
    params = SimpleNamespace(extra_args=extra)
    async for _ in plugin._patched_generate(engine, "p", params, "r"):  # type: ignore[arg-type]
        pass
    assert engine.calls == calls

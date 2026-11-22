"""Position-selective residual-stream capture (issue #40), CPU only: the selector rules,
the plugin's read and trim of a capture file, the wire round trip and the client encoding.
The value-level checks against a live engine are the GPU suites."""

from __future__ import annotations

import json
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import vllm_lens._activations_plugin as plugin
from vllm_lens._helpers._activation_store import ActivationStore
from vllm_lens._helpers._capture_file import (
    CAPTURE_DIR_ENV,
    HEADER_BYTES,
    capture_path,
    seal,
)
from vllm_lens._helpers._positions import (
    LAST_PROMPT_TOKEN,
    POSITIONS_KEY,
    POSITIONS_META_KEY,
    normalize_capture_positions,
    parse_capture_positions,
)
from vllm_lens._helpers._serialize import (
    decode_activations,
    serialize_activations,
    serialize_activations_binary,
    tensor_from_bytes,
)
from vllm_lens.client import VLLMLensClient

HIDDEN = 4


# ---------------------------------------------------------------------------
# Selector parsing and entry-point validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("null", None),
        (LAST_PROMPT_TOKEN, LAST_PROMPT_TOKEN),
        (json.dumps(LAST_PROMPT_TOKEN), LAST_PROMPT_TOKEN),
        ([5, 2, 5], [2, 5]),
        ((5, 2), [2, 5]),
        ("[5, 2, 5]", [2, 5]),
        ([0], [0]),
    ],
)
def test_parse_capture_positions(value, expected):
    assert parse_capture_positions(value) == expected


@pytest.mark.parametrize(
    "value",
    [[-1], [1.5], [True], ["3"], [], "last", "last token", "{}", {"a": 1}, 3, "3"],
)
def test_parse_rejects_invalid_selectors(value):
    with pytest.raises(ValueError, match=POSITIONS_KEY):
        parse_capture_positions(value)


def test_normalize_rewrites_extra_args_in_place():
    extra = {"output_residual_stream": [2], POSITIONS_KEY: "[5, 2, 5]"}
    normalize_capture_positions(extra)
    assert extra[POSITIONS_KEY] == [2, 5]
    extra = {
        "output_residual_stream": [2],
        POSITIONS_KEY: json.dumps(LAST_PROMPT_TOKEN),
    }
    normalize_capture_positions(extra)
    assert extra[POSITIONS_KEY] == LAST_PROMPT_TOKEN
    extra = {"output_residual_stream": [2], POSITIONS_KEY: None}
    normalize_capture_positions(extra)
    assert POSITIONS_KEY not in extra
    for untouched in (None, {}, {"output_residual_stream": [2]}):
        normalize_capture_positions(untouched)


def test_normalize_requires_residual_stream_capture():
    with pytest.raises(ValueError, match="requires output_residual_stream"):
        normalize_capture_positions({POSITIONS_KEY: LAST_PROMPT_TOKEN})


def test_trim_drops_rows_from_the_surplus_pass():
    acts = {
        "residual_stream": torch.arange(3.0).view(1, 3, 1),
        POSITIONS_META_KEY: [0, 9, 12],
    }
    plugin._trim_activations(acts, expected_len=12)
    assert acts["residual_stream"].flatten().tolist() == [0.0, 1.0]
    assert acts[POSITIONS_META_KEY] == [0, 9]
    kept = {"residual_stream": torch.zeros(1, 2, 1), POSITIONS_META_KEY: [0, 9]}
    rs = kept["residual_stream"]
    plugin._trim_activations(kept, expected_len=12)
    assert kept["residual_stream"] is rs and kept[POSITIONS_META_KEY] == [0, 9]
    plain = {"residual_stream": torch.zeros(1, 13, 1)}
    plugin._trim_activations(plain, expected_len=12)
    assert plain["residual_stream"].shape == (1, 12, 1)


def test_offline_entry_point_validates_before_any_rpc(stub_engine):
    calls: list[str] = []
    llm = stub_engine(lambda name, args=(): calls.append(name), offline=True)
    bad = SimpleNamespace(
        extra_args={"output_residual_stream": [2], POSITIONS_KEY: [-1]},
        skip_reading_prefix_cache=False,
    )
    with pytest.raises(ValueError, match=POSITIONS_KEY):
        plugin._prepare_offline_params(llm, [bad])  # type: ignore[arg-type]
    assert calls == []
    good = SimpleNamespace(
        extra_args={"output_residual_stream": [2], POSITIONS_KEY: "[5, 2]"},
        skip_reading_prefix_cache=False,
    )
    state = plugin._prepare_offline_params(llm, [good])  # type: ignore[arg-type]
    # Generation uses copies; the caller's params keep their values.
    (copied,) = state["params"]
    assert copied.extra_args[POSITIONS_KEY] == [2, 5]
    assert copied.skip_reading_prefix_cache is True
    assert good.extra_args[POSITIONS_KEY] == "[5, 2]"
    assert state["captures"] and calls == []


async def test_async_entry_point_attaches_trimmed_selection(
    monkeypatch, tmp_path, stub_engine
):
    """The final output gets the capture file the worker wrote, trimmed to the positions
    the request computed, and the file is removed."""
    monkeypatch.setenv(CAPTURE_DIR_ENV, str(tmp_path))
    module = ModuleType("vllm.v1.engine")
    module.EngineCoreRequest = type("EngineCoreRequest", (), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", module)
    calls: list[str] = []

    async def collective_rpc(name, args=()):
        calls.append(name)

    async def original_generate(self, prompt, sampling_params, request_id, **kwargs):
        key = sampling_params.extra_args["_capture_key"]
        path = capture_path(str(tmp_path), key, request_id, partial=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT)
        rows = torch.arange(3.0).view(3, 1, 1).to(torch.bfloat16)
        os.pwrite(fd, rows.view(torch.int16).numpy().tobytes(), HEADER_BYTES)
        seal(
            fd,
            path,
            dtype=torch.bfloat16,
            rows=3,
            layers=[2],
            hidden=1,
            positions=[0, 9, 12],
        )
        yield SimpleNamespace(
            finished=True,
            prompt_token_ids=[0] * 10,
            outputs=[SimpleNamespace(token_ids=[1, 2, 3])],
        )

    monkeypatch.setattr(plugin, "_original_generate", original_generate)
    engine = stub_engine(collective_rpc, offline=False)
    params = SimpleNamespace(
        extra_args={"output_residual_stream": [2], POSITIONS_KEY: "[12, 9, 0]"},
        skip_reading_prefix_cache=False,
    )
    outputs = [o async for o in plugin._patched_generate(engine, "hi", params, "req")]  # type: ignore[arg-type]
    # Generation uses a copy; the caller's params keep their value.
    assert params.extra_args[POSITIONS_KEY] == "[12, 9, 0]"
    acts = outputs[-1].activations
    # 10 prompt + 3 generated tokens compute positions 0..11; 12 is surplus.
    assert acts["residual_stream"].flatten().tolist() == [0.0, 1.0]
    assert acts[POSITIONS_META_KEY] == [0, 9]
    assert calls == [] and not list(tmp_path.iterdir())

    bad = SimpleNamespace(
        extra_args={POSITIONS_KEY: LAST_PROMPT_TOKEN}, skip_reading_prefix_cache=False
    )
    with pytest.raises(ValueError, match="requires output_residual_stream"):
        async for _ in plugin._patched_generate(engine, "hi", bad, "req"):  # type: ignore[arg-type]
            pass


# ---------------------------------------------------------------------------
# Wire format and client
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("transport", ["base64", "binary"])
def test_positions_survive_both_transports(transport):
    acts = {
        "residual_stream": torch.randn(2, 1, HIDDEN, dtype=torch.bfloat16),
        POSITIONS_META_KEY: [9],
    }
    store = ActivationStore()
    if transport == "binary":
        serialized = serialize_activations_binary(acts, store)
        assert "handle" in serialized["residual_stream"]
    else:
        serialized = serialize_activations(acts)
    response = {"activations": json.loads(json.dumps(serialized))}  # JSON-safe
    decoded = decode_activations(
        response,
        fetch_bytes=lambda h: store.get(h)[0],  # type: ignore[index]
    )
    assert torch.equal(decoded["residual_stream"], acts["residual_stream"])
    assert decoded[POSITIONS_META_KEY] == [9]
    if transport == "binary":
        raw, meta = store.get(serialized["residual_stream"]["handle"])  # type: ignore[misc]
        assert torch.equal(tensor_from_bytes(raw, meta), acts["residual_stream"])


def test_client_encodes_capture_positions():
    client = VLLMLensClient("http://localhost:8000")
    xargs = client._build_xargs([2], None, capture_positions=LAST_PROMPT_TOKEN)
    assert xargs == {"output_residual_stream": "[2]", POSITIONS_KEY: LAST_PROMPT_TOKEN}
    xargs = client._build_xargs([2], None, capture_positions=[5, 0])
    assert xargs[POSITIONS_KEY] == "[5, 0]"
    assert parse_capture_positions(xargs[POSITIONS_KEY]) == [0, 5]
    assert POSITIONS_KEY not in client._build_xargs([2], None)


# ---------------------------------------------------------------------------
# Fork: capture options at submission
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        (
            {"output_residual_stream": "[2]", POSITIONS_KEY: "[5, 2, 5]"},
            {"output_residual_stream": [2], POSITIONS_KEY: [2, 5]},
        ),
        (
            {"output_residual_stream": [2], POSITIONS_KEY: None},
            {"output_residual_stream": [2]},
        ),
    ],
    ids=["strings", "none"],
)
def test_capture_options_are_normalized(extra: dict, expected: dict) -> None:
    """The layer list and the selector replace their string forms; None is removed."""
    plugin._check_capture_options(extra)
    assert extra == expected

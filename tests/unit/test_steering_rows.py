"""Steering tables, steering rows, n > 1 and the add-only graph option, without vLLM or CUDA."""

import pickle
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import vllm_lens._activations_plugin as plugin
from vllm_lens import SteeringRow, SteeringVector
from vllm_lens._cudagraph import CONFIG_KEY, LensGraphConfig

TABLE = torch.arange(24, dtype=torch.float32).reshape(3, 8)


def extension(requests):
    """A CPU worker with table `t` and the given requests, ids mapped to extra_args."""
    return SimpleNamespace(
        model_runner=SimpleNamespace(
            model=torch.nn.Linear(8, 8, bias=False),
            model_config=SimpleNamespace(get_hidden_size=lambda: 8),
            requests={
                req_id: SimpleNamespace(
                    sampling_params=SimpleNamespace(extra_args=extra),
                    num_computed_tokens=0,
                )
                for req_id, extra in requests.items()
            },
            input_batch=SimpleNamespace(num_reqs=len(requests), req_ids=list(requests)),
        ),
        model_config=SimpleNamespace(get_total_num_hidden_layers=lambda: 4),
        _steering_data={},
        _steering_tables={"t": TABLE.clone()},
        _row_steering={},
    )


def row(index, layers, scale):
    """One steering row of table `t`, as the plugin passes it to the workers."""
    return {"table": "t", "row": index, "layer_indices": layers, "scale": scale}


def test_rows_sum_per_layer_and_are_dropped_with_their_request(worker, monkeypatch):
    """Each layer gets the sum of its rows; a finished request's rows are dropped."""
    ext = extension(
        {
            "a-0": {"apply_steering_rows": [row(1, [2, 3], 2.0)]},
            "b-0": {"apply_steering_rows": [row(0, [2], 1.0), row(2, [2], -0.5)]},
        }
    )
    monkeypatch.setattr(
        worker, "_batch_layout", lambda *a, **k: torch.tensor([0, 3, 5])
    )
    plan = worker._steer_pass(ext)
    assert [(s.start, s.end) for s in plan[2]] == [(0, 3), (3, 5)]
    assert [(s.start, s.end) for s in plan[3]] == [(0, 3)]
    torch.testing.assert_close(plan[2][0].data.add, 2.0 * TABLE[1:2])
    torch.testing.assert_close(plan[3][0].data.add, 2.0 * TABLE[1:2])
    torch.testing.assert_close(plan[2][1].data.add, TABLE[0:1] - 0.5 * TABLE[2:3])
    assert plan[2][0].data.norm is None
    del ext.model_runner.requests["a-0"]
    ext.model_runner.input_batch = SimpleNamespace(num_reqs=1, req_ids=["b-0"])
    monkeypatch.setattr(worker, "_batch_layout", lambda *a, **k: torch.tensor([0, 2]))
    worker._steer_pass(ext)
    assert set(ext._row_steering) == {"b-0"}


@pytest.mark.parametrize("req_id", ["r-abc", "0_r-abc", "7_r-abc"])
def test_child_requests_find_the_parent_vectors(worker, monkeypatch, req_id):
    """A child of an n > 1 request finds its parent's vectors through _steering_id."""
    ext = extension({req_id: {"_steering_id": "r"}})
    vector = SteeringVector(activations=torch.ones(1, 8), layer_indices=[1])
    worker.HiddenStatesExtension.set_steering_data(ext, "r", pickle.dumps([vector]))
    monkeypatch.setattr(worker, "_batch_layout", lambda *a, **k: torch.tensor([0, 4]))
    plan = worker._steer_pass(ext)
    torch.testing.assert_close(plan[1][0].data.add, torch.ones(1, 8))


def test_tables_are_set_and_cleared(worker):
    """A table is held as float32 under its name; clearing None drops every table."""
    ext = extension({})
    ext.install_hooks = lambda: None
    payload = plugin.serialize_tensor(TABLE.to(torch.bfloat16))
    worker.HiddenStatesExtension.set_steering_table(ext, "u", payload)
    assert ext._steering_tables["u"].dtype == torch.float32
    worker.HiddenStatesExtension.clear_steering_table(ext, "u")
    assert set(ext._steering_tables) == {"t"}
    worker.HiddenStatesExtension.clear_steering_table(ext, None)
    assert ext._steering_tables == {}


def _vllm_config(layers=4):
    """A VllmConfig stand-in with `layers` decoder layers."""
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(num_hidden_layers=layers)
        )
    )


@pytest.mark.parametrize(
    "rows,vectors,message",
    [
        ([SteeringRow(table="u", row=0, layer_indices=[0])], None, "not registered"),
        ([SteeringRow(table="t", row=3, layer_indices=[0])], None, "has 3 rows"),
        ([SteeringRow(table="t", row=0, layer_indices=[4])], None, "out of range"),
        (
            [SteeringRow(table="t", row=0, layer_indices=[0])],
            [SteeringVector(activations=torch.ones(1, 8), layer_indices=[0])],
            "cannot be combined",
        ),
    ],
    ids=["unknown-table", "row", "layer", "with-vectors"],
)
def test_invalid_rows_fail_at_submission(rows, vectors, message):
    """A bad row reference raises before the request reaches the engine."""
    with pytest.raises(ValueError, match=message):
        plugin._check_steering_rows(rows, {"t": 3}, vectors, _vllm_config())


def test_rows_decode_from_the_wire_format():
    """A JSON string of rows decodes and checks to plain dicts."""
    wire = '[{"table": "t", "row": 2, "layer_indices": [1, 3], "scale": -1.5}]'
    rows = plugin._decode_steering_rows(wire)
    assert rows is not None
    assert plugin._check_steering_rows(rows, {"t": 3}, None, _vllm_config()) == [
        row(2, [1, 3], -1.5)
    ]


@pytest.mark.parametrize(
    "n,extra,hooks,fails",
    [
        (1, {"output_residual_stream": [0]}, None, False),
        (2, {}, None, False),
        (2, {"output_residual_stream": [0]}, None, True),
        (2, {"output_qk": True}, None, True),
        (2, {}, ["hook"], True),
    ],
    ids=["one-capture", "many-steer", "many-capture", "many-qk", "many-hooks"],
)
def test_parallel_sampling_rejects_capture(n, extra, hooks, fails):
    """With n > 1, capture and hooks raise; steering alone passes."""
    params = SimpleNamespace(n=n)
    if fails:
        with pytest.raises(ValueError, match="n > 1"):
            plugin._reject_parallel_capture(params, extra, hooks)
    else:
        plugin._reject_parallel_capture(params, extra, hooks)


@pytest.mark.parametrize("vectors", [False, True])
async def test_async_generate_keys_vectors_by_request_id(monkeypatch, vectors):
    """Vectors are keyed by the request id in extra_args; a client-sent key is dropped."""
    received: list = []

    async def original(self, prompt, params, request_id, **kwargs):
        received.append(params)
        yield SimpleNamespace(finished=True)

    monkeypatch.setattr(plugin, "_original_generate", original)
    for name in ("vllm", "vllm.v1", "vllm.v1.engine"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["vllm.v1.engine"].EngineCoreRequest = type("EngineCoreRequest", (), {})  # type: ignore[attr-defined]

    async def collective_rpc(method: str, args: tuple = ()) -> list:
        return []

    engine = SimpleNamespace(
        vllm_config=SimpleNamespace(additional_config=None),
        _hooks_installed=True,
        _lens_projections=frozenset(),
        collective_rpc=collective_rpc,
    )
    extra: dict[str, object] = {"_steering_id": "other"}
    if vectors:
        extra["apply_steering_vectors"] = [
            SteeringVector(activations=torch.ones(1, 4), layer_indices=[0])
        ]
    params = SimpleNamespace(n=2, extra_args=extra, skip_reading_prefix_cache=False)
    async for _ in plugin._patched_generate(engine, "p", params, "r"):  # type: ignore[arg-type]
        pass
    (sent,) = received
    assert sent.extra_args.get("_steering_id") == ("r" if vectors else None)


@pytest.mark.parametrize(
    "stored,expected",
    [
        ({"steer_layers": [1]}, True),
        ({"steer_layers": [1], "steer_norm_match": False}, False),
        ({"steer_layers": [1], "steer_norm_match": "no"}, "must be a bool"),
    ],
    ids=["default", "off", "not-a-bool"],
)
def test_norm_match_option_is_read(stored, expected):
    """steer_norm_match defaults to true and must be a bool."""
    config = SimpleNamespace(additional_config={CONFIG_KEY: stored})
    if isinstance(expected, str):
        with pytest.raises(ValueError, match=expected):
            LensGraphConfig.from_vllm_config(config)
    else:
        assert LensGraphConfig.from_vllm_config(config).steer_norm_match is expected


@pytest.mark.parametrize(
    "vectors,rows,message",
    [
        (
            [
                SteeringVector(
                    activations=torch.ones(1, 8), layer_indices=[1], norm_match=True
                )
            ],
            None,
            "norm_match",
        ),
        (
            None,
            [SteeringRow(table="t", row=0, layer_indices=[2])],
            r"apply_steering_rows .*\[2\]",
        ),
    ],
    ids=["norm-match", "row-layer"],
)
def test_graph_mode_rejects_unserved_steering(vectors, rows, message):
    """Without the norm buffer norm_match fails; a row on an unarmed layer fails."""
    config = LensGraphConfig(steer_layers=(1,), steer_norm_match=False)
    with pytest.raises(ValueError, match=message):
        config.reject(vectors=vectors, rows=rows)

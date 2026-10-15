"""Steering lookup, the per-request row table, and the two paths that apply it. No GPU."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_lens._graph_worker import _GRAPH_STATE, _fill_steer_buffers, _GraphBuffers
from vllm_lens._helpers.types import SteeringVector
from vllm_lens._worker_ext import (
    _apply_steering,
    _apply_table_rows,
    _steering_key,
    _SteerTable,
)

HIDDEN = 8
LAYER = 2
MAX_TOKENS = 16


def _vector(seed: int, **kwargs) -> SteeringVector:
    """A reproducible random vector on ``LAYER``."""
    generator = torch.Generator().manual_seed(seed)
    return SteeringVector(
        activations=torch.randn(1, HIDDEN, generator=generator),
        layer_indices=[LAYER],
        **kwargs,
    )


@pytest.mark.parametrize(
    ("internal_id", "extra_args", "expected"),
    [
        ("0", {"_steering_id": "_steer_3"}, "_steer_3"),
        ("0", {"_steering_id": "_steer_9"}, None),
        ("cmpl-abc-0", None, "cmpl-abc"),
        ("cmpl-abc-0", {}, "cmpl-abc"),
        ("other-7", None, None),
    ],
    ids=["offline", "offline-unknown", "async-suffix", "async-empty", "miss"],
)
def test_lookup(internal_id: str, extra_args: dict | None, expected: str | None):
    """Offline ids resolve through the sentinel, async ids through the external id."""
    extension = SimpleNamespace(_steering_data={"_steer_3": [], "cmpl-abc": []})
    assert _steering_key(extension, internal_id, extra_args) == expected  # type: ignore[arg-type]


def test_table_rows_are_reused_and_grown():
    """Released rows are reused, the table doubles when full, and old rows survive growth."""
    table = _SteerTable(device=torch.device("cpu"), hidden=HIDDEN, capacity=2)
    first = _vector(0, scale=2.0)
    assert table.row("a", [first]) == 1
    assert table.row("a", [first]) == 1
    assert table.row("b", [_vector(1)]) == 2
    assert table.capacity == 4
    torch.testing.assert_close(table.add[LAYER][1], first.activations[0] * 2.0)
    table.release("a")
    assert table.row("c", [_vector(2)]) == 1
    assert table.row("d", [_vector(3, position_indices=None)]) == 3


@pytest.mark.parametrize(
    ("activations", "expected"),
    [(torch.ones(1, HIDDEN + 1), 0), (torch.ones(1, 2, HIDDEN), -1)],
    ids=["wrong-width", "position-specific"],
)
def test_a_vector_outside_the_table(activations: torch.Tensor, expected: int):
    """A bad vector gets row 0 (no steering); a position-specific one gets -1."""
    table = _SteerTable(device=torch.device("cpu"), hidden=HIDDEN)
    vector = SteeringVector(activations=activations, layer_indices=[LAYER])
    assert table.row("a", [vector]) == expected
    # A failed row goes back to the free list; no row stays taken.
    assert table.row("b", [_vector(0)]) == 1


@pytest.mark.parametrize("norm_match", [False, True])
@pytest.mark.parametrize("fused", [False, True], ids=["plain", "fused-residual"])
def test_table_rows_match_the_per_request_path(norm_match: bool, fused: bool):
    """Two steered requests around an unsteered one end up as the per-request path leaves
    them, to float32 rounding."""
    generator = torch.Generator().manual_seed(0)
    output = torch.randn(10, HIDDEN, generator=generator)
    residual = torch.randn(10, HIDDEN, generator=generator) if fused else None
    norm_ref = output + residual if residual is not None else output
    spans = [(0, 4), (6, 10)]
    configs = [
        _vector(n, scale=scale, norm_match=norm_match)
        for n, scale in [(1, 0.5), (2, -2.0)]
    ]
    expected = output.clone()
    for (start, end), config in zip(spans, configs):
        _apply_steering([config], LAYER, expected, start, end, 0, norm_ref)
    table = _SteerTable(device=torch.device("cpu"), hidden=HIDDEN)
    rows = [table.row(f"r{n}", [config]) for n, config in enumerate(configs)]
    batched = output.clone()
    _apply_table_rows(
        table,
        LAYER,
        [(row, *span) for row, span in zip(rows, spans)],
        batched,
        norm_ref,
    )
    torch.testing.assert_close(batched, expected, atol=1e-4, rtol=1e-4)
    assert torch.equal(batched[4:6], output[4:6])


def _fill(
    extension: SimpleNamespace,
    state: _GraphBuffers,
    bounds: list[int],
    keys: list[str | None],
    step: int = 0,
) -> None:
    """Run one graph fill with ``keys[i]`` as request i's sentinel and ``bounds`` as its
    layout. Request ids carry ``step``, as vLLM's do; the fill must not depend on it."""
    requests = {
        f"req-{step}-{i}": SimpleNamespace(
            sampling_params=SimpleNamespace(
                extra_args={"_steering_id": key} if key else {}
            ),
            num_computed_tokens=0,
        )
        for i, key in enumerate(keys)
    }
    extension.model_runner = SimpleNamespace(
        input_batch=SimpleNamespace(num_reqs=len(keys), req_ids=list(requests)),
        requests=requests,
        query_start_loc=SimpleNamespace(np=np.array(bounds, dtype=np.int32)),
        model=torch.nn.Linear(1, 1),
        model_config=SimpleNamespace(get_hidden_size=lambda: HIDDEN),
    )
    _fill_steer_buffers(extension, state, torch.tensor(bounds))


def _buffers() -> _GraphBuffers:
    """Graph buffers for one steered layer, on the CPU."""
    return _GraphBuffers(
        add=torch.zeros(1, MAX_TOKENS, HIDDEN),
        norm=torch.zeros(1, MAX_TOKENS, HIDDEN),
        capture=torch.zeros(0, MAX_TOKENS, HIDDEN),
        steer_slots={LAYER: 0},
        capture_slots={},
    )


def test_graph_rows_land_on_their_requests():
    """Two steered requests around an unsteered one fill their own rows; a reused key takes
    its new vector, and a step with no steering zeroes the last step's rows."""
    state = _buffers()
    added = _vector(0, scale=8.0)
    matched = _vector(1, scale=4.0, norm_match=True)
    extension = SimpleNamespace(
        _steering_data={"_steer_0": [added], "_steer_2": [matched]}, _steer_table=None
    )
    _fill(extension, state, [0, 4, 6, 7], ["_steer_0", None, "_steer_2"])
    direction = matched.activations[0]
    unit = direction / (direction.norm() + 1e-6)
    torch.testing.assert_close(
        state.add[0, 0:4], (added.activations * 8.0).expand(4, -1)
    )
    assert state.add[0, 4:].abs().sum() == 0
    torch.testing.assert_close(state.norm[0, 6:7], (unit * 4.0).unsqueeze(0))
    assert state.norm[0, :6].abs().sum() == 0 and state.dirty == 7

    # What set_steering_data does: new vectors under the key, and its row released.
    second = _vector(2, scale=1.0)
    extension._steering_data["_steer_0"] = [second]
    extension._steer_table.release("_steer_0")
    _fill(extension, state, [0, 2], ["_steer_0"], step=1)
    torch.testing.assert_close(state.add[0, 0:2], second.activations.expand(2, -1))

    _fill(extension, state, [0, 1], [None], step=2)
    assert state.add.abs().sum() == 0 and state.norm.abs().sum() == 0
    assert state.dirty == 0


def test_a_position_specific_vector_fills_its_positions():
    """A position-specific vector fills only the named positions of its request, the
    ``norm_match`` one as unit rows in the norm buffer; the cached rows serve a later step."""
    state = _buffers()
    vector = SteeringVector(
        activations=torch.ones(1, 2, HIDDEN),
        layer_indices=[LAYER],
        position_indices=[1, 5],
        scale=2.0,
        norm_match=True,
    )
    extension = SimpleNamespace(
        _steering_data={"_steer_0": [vector]}, _steer_table=None
    )
    for step in range(2):
        _fill(extension, state, [0, 4], ["_steer_0"], step=step)
        unit = torch.full((HIDDEN,), 2.0 / (HIDDEN**0.5 + 1e-6))
        torch.testing.assert_close(state.norm[0, 1], unit)
        assert state.norm[0, [0, 2, 3]].abs().sum() == 0 and state.add.abs().sum() == 0
    assert set(extension._steer_table.units) == {"_steer_0"}


@pytest.mark.parametrize("fused", [False, True], ids=["plain", "fused-residual"])
def test_the_lens_op_steers_then_captures(monkeypatch: pytest.MonkeyPatch, fused: bool):
    """The op adds ``add + norm * ||stream||`` to the first ``n`` rows only, and captures the
    steered stream; more rows than the buffers hold raise."""
    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(3, HIDDEN, generator=generator)
    residual = torch.randn(3, HIDDEN, generator=generator) if fused else None
    state = _GraphBuffers(
        add=torch.randn(1, 4, HIDDEN, generator=generator),
        norm=torch.randn(1, 4, HIDDEN, generator=generator),
        capture=torch.zeros(1, 4, HIDDEN),
        steer_slots={LAYER: 0},
        capture_slots={LAYER: 0},
    )
    monkeypatch.setitem(_GRAPH_STATE, "buffers", state)
    stream = hidden + residual if residual is not None else hidden
    expected = (
        hidden
        + state.add[0, :3]
        + state.norm[0, :3] * stream.norm(dim=-1, keepdim=True)
    )
    steered = hidden.clone()
    torch.ops.vllm_lens.lens(steered, residual, 0, 0)
    torch.testing.assert_close(steered, expected)
    captured = expected + residual if residual is not None else expected
    torch.testing.assert_close(state.capture[0, :3], captured)
    assert state.capture[0, 3].abs().sum() == 0
    with pytest.raises(RuntimeError, match="max_num_batched_tokens"):
        torch.ops.vllm_lens.lens(torch.zeros(5, HIDDEN), None, 0, -1)


def test_a_bad_position_specific_vector_zeroes_its_span_only():
    """A position-specific vector of the wrong width leaves its request unsteered and the
    other request's rows as they were."""
    state = _buffers()
    good = _vector(0, scale=3.0)
    bad = SteeringVector(
        activations=torch.ones(1, 2, HIDDEN + 1),
        layer_indices=[LAYER],
        position_indices=[0, 1],
    )
    extension = SimpleNamespace(
        _steering_data={"_steer_0": [good], "_steer_1": [bad]}, _steer_table=None
    )
    _fill(extension, state, [0, 2, 4], ["_steer_0", "_steer_1"])
    torch.testing.assert_close(
        state.add[0, 0:2], (good.activations * 3.0).expand(2, -1)
    )
    assert state.add[0, 2:4].abs().sum() == 0 and state.norm.abs().sum() == 0


def test_clearing_a_request_releases_its_table_rows():
    """``clear_request_state`` gives a key's table row back and drops its cached
    position-specific rows."""
    from vllm_lens._worker_ext import HiddenStatesExtension

    table = _SteerTable(device=torch.device("cpu"), hidden=HIDDEN)
    positional = SteeringVector(
        activations=torch.ones(1, 2, HIDDEN), layer_indices=[LAYER]
    )
    data = {"a": [_vector(0)], "p": [positional]}
    assert table.row("a", data["a"]) == 1 and table.row("p", data["p"]) == -1
    table.positional("p", data["p"])
    worker = HiddenStatesExtension.__new__(HiddenStatesExtension)
    worker._steering_data, worker._hook_data, worker._steer_table = data, {}, table
    worker.clear_request_state(["a", "p"], [])
    assert worker._steering_data == {} and table.free == [1]
    assert table.rows == {} and table.units == {}


def _graph_load(monkeypatch: pytest.MonkeyPatch, worker: SimpleNamespace) -> None:
    """Run the patched ``Worker.load_model`` on ``worker``, with vLLM's worker classes stubbed."""
    import sys
    import types

    import vllm_lens._graph_worker as graph_worker

    stub_worker = type("Worker", (), {"load_model": lambda self: None})
    stub_runner = type("GPUModelRunner", (), {"_model_forward": lambda self: None})
    for name, cls in (
        ("vllm.v1.worker.gpu_worker", stub_worker),
        ("vllm.v1.worker.gpu_model_runner", stub_runner),
    ):
        module = types.ModuleType(name)
        setattr(module, cls.__name__, cls)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(graph_worker, "_enable_graph_buffers", lambda *args: None)
    graph_worker.install_graph_patches()
    stub_worker.load_model(worker)


@pytest.mark.parametrize(
    ("change", "error", "match"),
    [
        ({"extension": False}, RuntimeError, "worker extension"),
        ({"layer": 40}, ValueError, "layers 0..23"),
        ({"pp": 2}, RuntimeError, "pipeline parallelism"),
        ({"dbo": True}, RuntimeError, "micro-batching"),
        ({"sp": True}, RuntimeError, "sequence parallelism"),
    ],
    ids=["foreign-extension", "layer-out-of-range", "pp", "dbo", "sp"],
)
def test_the_graph_mode_refuses_what_it_cannot_serve(
    monkeypatch: pytest.MonkeyPatch, change: dict, error: type, match: str
):
    """The load fails before any buffer is made, naming what is not served."""
    from vllm_lens._cudagraph import LensGraphConfig

    config = LensGraphConfig(steer_layers=(change.get("layer", LAYER),))
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            additional_config=config.to_additional_config(None),
            compilation_config=SimpleNamespace(
                pass_config=SimpleNamespace(enable_sp=change.get("sp", False))
            ),
        ),
        model_config=SimpleNamespace(get_total_num_hidden_layers=lambda: 24),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=change.get("pp", 1),
            use_ubatching=change.get("dbo", False),
        ),
    )
    if change.get("extension", True):
        worker._reset_state = lambda: None
    with pytest.raises(error, match=match):
        _graph_load(monkeypatch, worker)

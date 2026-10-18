"""Request payloads, the per-payload row table, and the paths that apply it. No GPU."""

import pickle
import sys
from dataclasses import asdict
from types import ModuleType, SimpleNamespace

import cloudpickle
import numpy as np
import pytest
import torch

import vllm_lens._graph_worker as graph_worker
from vllm_lens import Hook, SteeringVector
from vllm_lens._graph_worker import _fill_steer_buffers, _GraphBuffers
from vllm_lens._helpers.types import HOOKS_PAYLOAD, STEERING_PAYLOAD
from vllm_lens._worker_ext import (
    _apply_steering,
    _apply_table_rows,
    _release_payloads,
    _request_payload,
    _SteerTable,
    _token_layout,
)

HIDDEN = 8
LAYER = 2
MAX_TOKENS = 16


def _extension(table: _SteerTable | None = None) -> SimpleNamespace:
    """An extension with a float16 CPU model."""
    model = torch.nn.Linear(1, 1, dtype=torch.float16)
    return SimpleNamespace(
        model_runner=SimpleNamespace(model=model),
        _payloads={},
        _request_payloads={},
        _steer_table=table,
    )


def _state(extra: dict | None) -> SimpleNamespace:
    """A runner request state with ``extra`` as its extra_args."""
    return SimpleNamespace(sampling_params=SimpleNamespace(extra_args=extra))


def _vector(seed: int, **kwargs) -> SteeringVector:
    """A reproducible random vector on ``LAYER``."""
    generator = torch.Generator().manual_seed(seed)
    return SteeringVector(
        activations=torch.randn(1, HIDDEN, generator=generator),
        layer_indices=[LAYER],
        **kwargs,
    )


@pytest.mark.parametrize("key", [STEERING_PAYLOAD, HOOKS_PAYLOAD])
def test_requests_share_one_decoded_payload(key: str) -> None:
    """Two requests with the same bytes share one copy, freed when both have left."""
    if key == STEERING_PAYLOAD:
        raw = pickle.dumps([_vector(0)])
    else:
        raw = cloudpickle.dumps([Hook(fn=lambda ctx, h: None, layer_indices=[0])])
    extension = _extension()
    first = _request_payload(extension, _state({key: raw}), "a-0", key)  # type: ignore[arg-type]
    second = _request_payload(extension, _state({key: raw}), "b-0", key)  # type: ignore[arg-type]
    assert first is second and len(first) == 1
    if key == STEERING_PAYLOAD:
        assert first[0].activations.dtype == torch.float16
    _release_payloads(extension, "a-0")  # type: ignore[arg-type]
    assert len(extension._payloads) == 1
    _release_payloads(extension, "b-0")  # type: ignore[arg-type]
    assert extension._payloads == {} and extension._request_payloads == {}


@pytest.mark.parametrize(
    "state",
    [None, _state(None), _state({}), _state({STEERING_PAYLOAD: b"not a pickle"})],
    ids=["no-state", "no-extra", "empty", "bad-bytes"],
)
def test_missing_or_bad_payload_steers_nothing(state: SimpleNamespace | None) -> None:
    """A request without a payload, or with one that cannot be decoded, gets []."""
    extension = _extension()
    assert _request_payload(extension, state, "a-0", STEERING_PAYLOAD) == []  # type: ignore[arg-type]
    _release_payloads(extension, "a-0")  # type: ignore[arg-type]
    assert extension._payloads == {}


def test_the_last_request_of_a_payload_frees_its_row() -> None:
    """A payload's table row is given back when its last request leaves the runner."""
    table = _SteerTable(device=torch.device("cpu"), hidden=HIDDEN)
    extension = _extension(table)
    state = _state({STEERING_PAYLOAD: pickle.dumps([_vector(0)])})
    for req_id in ("a-0", "b-0"):
        configs = _request_payload(extension, state, req_id, STEERING_PAYLOAD)  # type: ignore[arg-type]
        digest = extension._request_payloads[(STEERING_PAYLOAD, req_id)]
        assert table.row(digest, configs) == 1
    _release_payloads(extension, "a-0")  # type: ignore[arg-type]
    assert table.free == []
    _release_payloads(extension, "b-0")  # type: ignore[arg-type]
    assert table.free == [1] and table.rows == {}


def test_table_rows_are_reused_and_grown():
    """Released rows are reused, the table doubles when full, and old rows survive growth."""
    table = _SteerTable(device=torch.device("cpu"), hidden=HIDDEN, capacity=2)
    first = _vector(0, scale=2.0)
    assert table.row(b"a", [first]) == 1
    assert table.row(b"a", [first]) == 1
    assert table.row(b"b", [_vector(1)]) == 2
    assert table.capacity == 4
    torch.testing.assert_close(table.add[LAYER][1], first.activations[0] * 2.0)
    table.release(b"a")
    assert table.row(b"c", [_vector(2)]) == 1
    assert table.row(b"d", [_vector(3)]) == 3


@pytest.mark.parametrize(
    ("activations", "expected"),
    [(torch.ones(1, HIDDEN + 1), 0), (torch.ones(1, 2, HIDDEN), -1)],
    ids=["wrong-width", "position-specific"],
)
def test_a_vector_outside_the_table(activations: torch.Tensor, expected: int):
    """A bad vector gets row 0 (no steering); a position-specific one gets -1."""
    table = _SteerTable(device=torch.device("cpu"), hidden=HIDDEN)
    vector = SteeringVector(activations=activations, layer_indices=[LAYER])
    assert table.row(b"a", [vector]) == expected
    # A failed row goes back to the free list; no row stays taken.
    assert table.row(b"b", [_vector(0)]) == 1


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
    rows = [table.row(bytes([n]), [config]) for n, config in enumerate(configs)]
    batched = output.clone()
    layout = _token_layout(spans, rows, torch.device("cpu"))
    _apply_table_rows(table, LAYER, layout, batched, norm_ref)
    torch.testing.assert_close(batched, expected, atol=1e-4, rtol=1e-4)
    assert torch.equal(batched[4:6], output[4:6])


def _fill(
    extension: SimpleNamespace,
    state: _GraphBuffers,
    bounds: list[int],
    vectors: list[list[SteeringVector] | None],
    step: int = 0,
) -> None:
    """Run one graph fill with ``vectors[i]`` as request i's vectors and ``bounds`` as its
    layout. Request ids carry ``step``, as vLLM's do; the fill must not depend on it."""
    requests = {
        f"req-{step}-{i}": SimpleNamespace(
            sampling_params=SimpleNamespace(
                extra_args={STEERING_PAYLOAD: pickle.dumps(v)} if v else {}
            ),
            num_computed_tokens=0,
        )
        for i, v in enumerate(vectors)
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
    """Two steered requests around an unsteered one fill their own rows, and a step with no
    steering zeroes the last step's rows."""
    state = _buffers()
    added = _vector(0, scale=8.0)
    matched = _vector(1, scale=4.0, norm_match=True)
    extension = _extension()
    _fill(extension, state, [0, 4, 6, 7], [[added], None, [matched]])
    direction = matched.activations[0]
    unit = direction / (direction.norm() + 1e-6)
    torch.testing.assert_close(
        state.add[0, 0:4], (added.activations * 8.0).expand(4, -1)
    )
    assert state.add[0, 4:].abs().sum() == 0
    torch.testing.assert_close(state.norm[0, 6:7], (unit * 4.0).unsqueeze(0))
    assert state.norm[0, :6].abs().sum() == 0 and state.dirty == 7

    _fill(extension, state, [0, 1], [None], step=2)
    assert state.add.abs().sum() == 0 and state.norm.abs().sum() == 0
    assert state.dirty == 0


def test_a_position_specific_vector_fills_its_positions():
    """A position-specific vector fills only the named positions of its request, the
    ``norm_match`` one as unit rows in the norm buffer, on every step."""
    state = _buffers()
    vector = SteeringVector(
        activations=torch.ones(1, 2, HIDDEN),
        layer_indices=[LAYER],
        position_indices=[1, 5],
        scale=2.0,
        norm_match=True,
    )
    extension = _extension()
    for step in range(2):
        _fill(extension, state, [0, 4], [[vector]], step=step)
        unit = torch.full((HIDDEN,), 2.0 / (HIDDEN**0.5 + 1e-6))
        torch.testing.assert_close(state.norm[0, 1], unit)
        assert state.norm[0, [0, 2, 3]].abs().sum() == 0 and state.add.abs().sum() == 0


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
    monkeypatch.setattr(graph_worker, "_BUFFERS", state)
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


@pytest.mark.parametrize(
    ("dtype", "magnitude"),
    [(torch.bfloat16, 256), (torch.float16, 2048), (torch.float32, 2**24)],
)
@pytest.mark.parametrize("fused", [False, True], ids=["plain", "fused-residual"])
@pytest.mark.parametrize("steer", [False, True], ids=["capture", "steer-and-capture"])
def test_op_captures_the_actual_output(monkeypatch, dtype, magnitude, fused, steer):
    """Capture follows the rounded, mutated output, including residual cancellation."""
    hidden = torch.full((1, HIDDEN), magnitude, dtype=dtype)
    residual = -hidden.clone() if fused else None
    expected_hidden = hidden + 1 if steer else hidden.clone()
    state = _GraphBuffers(
        add=torch.ones(1, 2, HIDDEN, dtype=dtype),
        norm=torch.zeros(1, 2, HIDDEN, dtype=dtype),
        capture=torch.zeros(1, 2, HIDDEN, dtype=dtype),
        steer_slots={LAYER: 0},
        capture_slots={LAYER: 0},
    )
    monkeypatch.setattr(graph_worker, "_BUFFERS", state)

    torch.ops.vllm_lens.lens(hidden, residual, 0 if steer else -1, 0)

    assert torch.equal(hidden, expected_hidden)
    expected = hidden + residual if fused else hidden
    assert torch.equal(state.capture[0, :1], expected)
    assert state.capture[0, 1].count_nonzero() == 0


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
    extension = _extension()
    _fill(extension, state, [0, 2, 4], [[good], [bad]])
    torch.testing.assert_close(
        state.add[0, 0:2], (good.activations * 3.0).expand(2, -1)
    )
    assert state.add[0, 2:4].abs().sum() == 0 and state.norm.abs().sum() == 0


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
    from vllm_lens._cudagraph import CONFIG_KEY, LensGraphConfig

    config = LensGraphConfig(steer_layers=(change.get("layer", LAYER),))
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            additional_config={CONFIG_KEY: asdict(config)},
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
    stub_worker = type("Worker", (), {"load_model": lambda self: None})
    stub_runner = type("GPUModelRunner", (), {"_model_forward": lambda self: None})
    for name, cls in (
        ("vllm.v1.worker.gpu_worker", stub_worker),
        ("vllm.v1.worker.gpu_model_runner", stub_runner),
    ):
        module = ModuleType(name)
        setattr(module, cls.__name__, cls)
        monkeypatch.setitem(sys.modules, name, module)
    graph_worker.install_graph_patches()
    with pytest.raises(error, match=match):
        stub_worker.load_model(worker)

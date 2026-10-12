"""The CUDA-graph buffer fill writes each request's rows from the per-key table. No GPU."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm_lens._worker_ext as worker_ext
from vllm_lens import SteeringVector
from vllm_lens._cudagraph import _STEER_STATE, _SteerBuffers, _fill_steer_buffers

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


def _step(
    monkeypatch: pytest.MonkeyPatch,
    extension: SimpleNamespace,
    bounds: list[int],
    keys: list[str | None],
    cpu_layout: bool,
) -> None:
    """Run one fill with ``keys[i]`` as request i's sentinel and ``bounds`` as its layout."""
    layout = torch.tensor(bounds, dtype=torch.int32)
    requests = {
        f"req-{i}": SimpleNamespace(
            sampling_params=SimpleNamespace(
                extra_args={"_steering_id": key} if key else {}
            ),
            num_computed_tokens=0,
        )
        for i, key in enumerate(keys)
    }
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(num_reqs=len(keys), req_ids=list(requests)),
        requests=requests,
    )
    if cpu_layout:
        runner.query_start_loc = SimpleNamespace(np=np.array(bounds, dtype=np.int32))
    extension.model_runner = runner
    monkeypatch.setattr(worker_ext, "_batch_layout", lambda runner, warn: layout)
    _fill_steer_buffers(extension)


@pytest.mark.parametrize("cpu_layout", [True, False], ids=["cpu-layout", "host-sync"])
def test_rows_land_on_their_requests(monkeypatch: pytest.MonkeyPatch, cpu_layout: bool):
    """Two steered requests around an unsteered one fill their own rows and no other."""
    state = _SteerBuffers(
        add=torch.zeros(1, MAX_TOKENS, HIDDEN),
        norm=torch.zeros(1, MAX_TOKENS, HIDDEN),
        slots={LAYER: 0},
    )
    monkeypatch.setitem(_STEER_STATE, "buffers", state)
    added = _vector(0, scale=8.0)
    matched = _vector(1, scale=4.0, norm_match=True)
    extension = SimpleNamespace(
        _steering_data={"_steer_0": [added], "_steer_2": [matched]}, _steer_rows={}
    )

    _step(
        monkeypatch, extension, [0, 4, 6, 7], ["_steer_0", None, "_steer_2"], cpu_layout
    )
    direction = matched.activations[0]
    unit = direction / (direction.norm() + 1e-6)
    torch.testing.assert_close(
        state.add[0, 0:4], (added.activations * 8.0).expand(4, -1)
    )
    assert state.add[0, 4:].abs().sum() == 0
    torch.testing.assert_close(state.norm[0, 6:7], (unit * 4.0).unsqueeze(0))
    assert state.norm[0, :6].abs().sum() == 0 and state.norm[0, 7:].abs().sum() == 0
    assert state.dirty == 7
    assert set(extension._steer_rows) == {"_steer_0", "_steer_2"}

    # The next step steers nothing, so the rows of the last step are zero again.
    _step(monkeypatch, extension, [0, 1], [None], cpu_layout)
    assert state.add.abs().sum() == 0 and state.norm.abs().sum() == 0
    assert state.dirty == 0


def test_a_bad_vector_steers_nothing(monkeypatch: pytest.MonkeyPatch):
    """A vector of the wrong width is logged once and its request gets zero rows."""
    state = _SteerBuffers(
        add=torch.zeros(1, MAX_TOKENS, HIDDEN),
        norm=torch.zeros(1, MAX_TOKENS, HIDDEN),
        slots={LAYER: 0},
    )
    monkeypatch.setitem(_STEER_STATE, "buffers", state)
    bad = SteeringVector(activations=torch.ones(1, HIDDEN + 1), layer_indices=[LAYER])
    extension = SimpleNamespace(_steering_data={"_steer_0": [bad]}, _steer_rows={})
    _step(monkeypatch, extension, [0, 3], ["_steer_0"], True)
    assert state.add.abs().sum() == 0 and state.norm.abs().sum() == 0
    assert extension._steer_rows["_steer_0"] is not None

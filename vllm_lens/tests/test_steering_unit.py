"""Request payloads decoded once and shared by requests with the same bytes. No GPU."""

import pickle
from types import SimpleNamespace

import cloudpickle
import pytest
import torch

from vllm_lens import Hook, SteeringVector
from vllm_lens._helpers.types import HOOKS_PAYLOAD, STEERING_PAYLOAD
from vllm_lens._worker_ext import _release_payloads, _request_payload


def _extension() -> SimpleNamespace:
    """An extension with a float16 CPU model."""
    model = torch.nn.Linear(1, 1, dtype=torch.float16)
    return SimpleNamespace(
        model_runner=SimpleNamespace(model=model), _payloads={}, _request_payloads={}
    )


def _state(extra: dict | None) -> SimpleNamespace:
    """A runner request state with ``extra`` as its extra_args."""
    return SimpleNamespace(sampling_params=SimpleNamespace(extra_args=extra))


@pytest.mark.parametrize("key", [STEERING_PAYLOAD, HOOKS_PAYLOAD])
def test_requests_share_one_decoded_payload(key: str) -> None:
    """Two requests with the same bytes share one copy, freed when both have left."""
    if key == STEERING_PAYLOAD:
        raw = pickle.dumps(
            [SteeringVector(activations=torch.ones(1, 4), layer_indices=[0])]
        )
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

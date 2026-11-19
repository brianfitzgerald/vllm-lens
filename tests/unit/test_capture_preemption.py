"""Capture retains the latest row for each absolute position."""

from types import SimpleNamespace

import torch

from vllm_lens._activations_plugin import _trim_activations


def test_capture_after_preemption(worker):
    """Recomputed positions replace earlier rows without removing later positions."""
    extra = {"output_residual_stream": [0]}
    request = SimpleNamespace(
        sampling_params=SimpleNamespace(extra_args=extra),
        num_computed_tokens=0,
        num_prompt_tokens=4,
    )
    ext = SimpleNamespace(
        model_runner=SimpleNamespace(
            input_batch=SimpleNamespace(num_reqs=1, req_ids=["r-0"]),
            requests={"r-0": request},
        ),
        _captured_states={},
        _captured_positions={},
        _selected_capture=set(),
        _pooled={},
    )
    stream = torch.arange(10).reshape(5, 2).float()
    for first, count in [(0, 2), (0, 2), (2, 2), (4, 1)]:
        request.num_computed_tokens = first
        worker._capture_rows(
            ext, 0, stream[first : first + count], torch.tensor([0, count])
        )
    acts = worker.HiddenStatesExtension._build_payload(ext, "r-0")["activations"]
    _trim_activations(acts, 5)
    torch.testing.assert_close(acts["residual_stream"][0], stream)
    assert "residual_stream_positions" not in acts

"""HookContext.position_offset across prefill chunks and decode."""

from types import SimpleNamespace

import numpy as np
import torch

from vllm_lens import Hook


class DecoderLayer(torch.nn.Module):
    """A decoder layer that returns its input."""

    def forward(self, positions, hidden_states, residual=None):
        """Return the hidden states and residual unchanged."""
        return hidden_states, residual


def test_position_offset_follows_computed_tokens(worker, monkeypatch) -> None:
    """Pre- and post-hooks see each slice's absolute start position."""
    context = SimpleNamespace(
        attn_metadata={"attn": SimpleNamespace(query_start_loc=object())}
    )
    monkeypatch.setattr(worker, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(worker, "get_forward_context", lambda: context)
    monkeypatch.setenv("VLLM_LENS_LAYER_PATH", "layers.{i}")
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([DecoderLayer()])
    request = SimpleNamespace(
        num_computed_tokens=0,
        sampling_params=SimpleNamespace(extra_args={"_hook_id": "h"}),
    )
    ext = worker.HiddenStatesExtension()
    ext.model_runner = SimpleNamespace(
        model=model,
        _model_forward=lambda *args, **kwargs: None,
        input_batch=SimpleNamespace(num_reqs=1, req_ids=["r-0"]),
        requests={"r-0": request},
    )
    ext.model_config = SimpleNamespace(
        get_total_num_hidden_layers=lambda: 1, get_num_layers=lambda _: 1
    )
    ext.parallel_config = SimpleNamespace(tensor_parallel_size=1)
    ext.rank = 0
    ext.install_hooks()
    seen = []
    ext._hook_data = {
        "h": [
            Hook(
                fn=lambda ctx, h: seen.append(("pre", ctx.position_offset)),
                layer_indices=[0],
                pre=True,
            ),
            Hook(
                fn=lambda ctx, h: seen.append(("post", ctx.position_offset)),
                layer_indices=[0],
            ),
        ]
    }
    # A 6-token prompt in chunks of 4 and 2, then one decode step.
    for chunk in (4, 2, 1):
        ext.model_runner.query_start_loc = SimpleNamespace(np=np.array([0, chunk]))
        model.layers[0](torch.arange(chunk), torch.zeros(chunk, 2))
        request.num_computed_tokens += chunk
    assert seen == [
        ("pre", 0),
        ("post", 0),
        ("pre", 4),
        ("post", 4),
        ("pre", 6),
        ("post", 6),
    ]

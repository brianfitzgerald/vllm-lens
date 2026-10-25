"""CUDA graph token buffers cover all positions and clear stale additions."""

import pytest
import torch
from vllm import SamplingParams

from vllm_lens import SteeringVector

from .conftest import lens_engine


@pytest.fixture(scope="module")
def engine():
    """Use a prefill large enough to exceed the old scratch capacity."""
    yield from lens_engine(
        {"steer_layers": [2], "capture_layers": [2]},
        max_num_batched_tokens=1024,
        enable_chunked_prefill=True,
    )


def test_large_position_set_and_buffer_cleanup(engine):
    """All 700 selected positions steer, and later requests remain unsteered."""
    hidden = engine.llm_engine.vllm_config.model_config.get_hidden_size()
    direction = torch.randn(hidden, generator=torch.Generator().manual_seed(0))
    vector = SteeringVector(
        activations=direction.expand(1, 700, -1).clone(),
        layer_indices=[2],
        norm_match=True,
        scale=4,
        position_indices=list(range(700)),
    )
    prompt = {"prompt_token_ids": list(range(1000, 1768))}
    params = SamplingParams(
        temperature=0,
        max_tokens=4,
        ignore_eos=True,
        extra_args={"apply_steering_vectors": [vector], "output_residual_stream": [2]},
    )
    output = engine.generate([prompt], params)[0]
    assert len(output.prompt_token_ids) > 700
    rows = output.activations["residual_stream"][0, :700].float()
    cosine = torch.nn.functional.cosine_similarity(rows, direction[None])
    assert (cosine > 0.9).all()
    ordinary = SamplingParams(
        temperature=0,
        max_tokens=4,
        ignore_eos=True,
        extra_args={"output_residual_stream": [2]},
    )
    first = engine.generate(["The future of AI is"], ordinary)[0]
    engine.generate([prompt], params)
    after = engine.generate(["The future of AI is"], ordinary)[0]
    assert first.outputs[0].token_ids == after.outputs[0].token_ids
    torch.testing.assert_close(
        first.activations["residual_stream"], after.activations["residual_stream"]
    )

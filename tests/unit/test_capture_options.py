"""Test that capture options are checked once when a request is submitted. No vLLM."""

import json
from types import SimpleNamespace

import pytest

import vllm_lens._activations_plugin as plugin


@pytest.mark.parametrize(
    "value,expected",
    [("[1, 3]", [1, 3]), ([2], [2]), (True, True), ("true", True)],
    ids=["json-list", "list", "bool", "json-bool"],
)
def test_layer_list_is_parsed(value, expected) -> None:
    """Test that a JSON-string layer list becomes the decoded value."""
    extra = {"output_residual_stream": value}
    plugin._check_capture_options(extra, set())
    assert extra["output_residual_stream"] == expected


def test_invalid_json_raises() -> None:
    """Test that an invalid layer list fails at submission."""
    with pytest.raises(json.JSONDecodeError):
        plugin._check_capture_options({"output_residual_stream": "[1,"}, set())


def test_offline_params_carry_the_parsed_list() -> None:
    """Test that the params sent to vLLM hold the list and the caller's keep the string."""
    llm = SimpleNamespace(
        _hooks_installed=True,
        _lens_projections=frozenset(),
        llm_engine=SimpleNamespace(vllm_config=SimpleNamespace(additional_config=None)),
        collective_rpc=lambda method, args=(): [],
    )
    params = SimpleNamespace(
        extra_args={"output_residual_stream": "[0, 2]"}, skip_reading_prefix_cache=False
    )
    sent = plugin._prepare_offline_params(llm, params)["params"]  # type: ignore[arg-type]
    assert sent.extra_args["output_residual_stream"] == [0, 2]
    assert params.extra_args["output_residual_stream"] == "[0, 2]"

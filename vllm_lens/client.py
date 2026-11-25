"""HTTP client for vllm-lens, mirroring the LLM interface.

Wraps the vLLM OpenAI-compatible API and the vllm-lens endpoints,
handling serialization of steering vectors and activations.

Usage::

    from vllm_lens.client import VLLMLensClient

    client = VLLMLensClient("http://localhost:8000")

    # Activation capture
    output = client.generate("Hello", max_tokens=10, capture_layers=[15])
    print(output.activations)

    # Steering
    output = client.generate("Hello", max_tokens=10, steering_vectors=[sv])
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import requests

from vllm_lens._helpers._serialize import (
    decode_activations,
)
from vllm_lens._helpers.types import SteeringVector


@dataclass
class GenerateOutput:
    """Output from a generate call, mirroring relevant fields of vLLM's RequestOutput."""

    text: str
    """Generated text."""

    activations: dict[str, Any] | None = None
    """Captured activations (residual stream, and its position metadata under a
    position selector), if requested."""

    logprobs: dict[str, Any] | None = None
    """Log-probabilities, if requested."""

    raw: dict[str, Any] = field(default_factory=dict, repr=False)
    """The full raw JSON response from the server."""


class VLLMLensClient:
    """HTTP client for a vLLM server with vllm-lens installed.

    Provides the same interface as the patched ``LLM`` class but
    communicates over HTTP.
    """

    def __init__(
        self,
        base_url: str,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float | None = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._session = requests.Session()
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"

    @property
    def model(self) -> str:
        """The model name served by the server."""
        if self._model is None:
            resp = self._session.get(
                f"{self.base_url}/v1/models", timeout=self._timeout
            ).json()
            self._model = resp["data"][0]["id"]
        return self._model

    def _build_xargs(
        self,
        capture_layers: list[int] | None,
        steering_vectors: list[SteeringVector] | None,
        activations_transport: str | None = None,
    ) -> dict[str, str]:
        xargs: dict[str, str] = {}
        if capture_layers is not None:
            xargs["output_residual_stream"] = json.dumps(capture_layers)
        if activations_transport is not None:
            xargs["activations_transport"] = activations_transport
        if steering_vectors is not None:
            xargs["apply_steering_vectors"] = json.dumps(
                [sv.model_dump() for sv in steering_vectors]
            )
        return xargs

    def _fetch_activation_bytes(self, handle: str) -> bytes:
        """Fetch a binary activation payload by handle (issue #31 transport).

        Used as the ``fetch_bytes`` callback for :func:`decode_activations` when
        a response carries an ``activations_transport="binary"`` handle instead
        of inline base64.
        """
        resp = self._session.get(
            f"{self.base_url}/v1/activations/{handle}", timeout=self._timeout
        )
        resp.raise_for_status()
        return resp.content

    def _parse_response(self, resp: dict[str, Any]) -> GenerateOutput:
        if "error" in resp:
            error = resp["error"]
            message = error.get("message", error) if isinstance(error, dict) else error
            raise RuntimeError(str(message))

        choice = resp["choices"][0]
        text = choice.get("message", {}).get("content") or choice.get("text", "")

        activations = (
            decode_activations(resp, fetch_bytes=self._fetch_activation_bytes) or None
        )

        lp = choice.get("logprobs")

        return GenerateOutput(
            text=text,
            activations=activations,
            logprobs=lp,
            raw=resp,
        )

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 16,
        temperature: float = 0.0,
        capture_layers: list[int] | None = None,
        steering_vectors: list[SteeringVector] | None = None,
        activations_transport: str | None = None,
        logprobs: int | None = None,
        echo: bool = False,
        **kwargs: Any,
    ) -> GenerateOutput:
        """Generate a completion from a raw prompt.

        Uses ``/v1/completions``. For chat messages, use :meth:`chat`.

        ``activations_transport="binary"`` returns captured activations as a
        handle fetched from ``GET /v1/activations/{handle}`` (no base64); the
        default inlines them as base64 in the JSON response.
        ``activations_transport="file"`` returns the path of the capture file the
        server wrote, which works only on the server's host; the client maps it
        and the caller removes it. Decoding is transparent in every case.
        """
        body: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }
        if logprobs is not None:
            body["logprobs"] = logprobs
        if echo:
            body["echo"] = True

        xargs = self._build_xargs(
            capture_layers,
            steering_vectors,
            activations_transport,
        )
        if xargs:
            body["vllm_xargs"] = xargs

        resp = self._session.post(
            f"{self.base_url}/v1/completions", json=body, timeout=self._timeout
        ).json()
        return self._parse_response(resp)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 16,
        temperature: float = 0.0,
        capture_layers: list[int] | None = None,
        steering_vectors: list[SteeringVector] | None = None,
        activations_transport: str | None = None,
        **kwargs: Any,
    ) -> GenerateOutput:
        """Generate a chat completion from a list of messages.

        Uses ``/v1/chat/completions``. See :meth:`generate` for
        ``activations_transport``.

        Example::

            output = client.chat([
                {"role": "user", "content": "What is the capital of France?"}
            ], capture_layers=[15])
        """
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }

        xargs = self._build_xargs(
            capture_layers,
            steering_vectors,
            activations_transport,
        )
        if xargs:
            body["vllm_xargs"] = xargs

        resp = self._session.post(
            f"{self.base_url}/v1/chat/completions", json=body, timeout=self._timeout
        ).json()
        return self._parse_response(resp)

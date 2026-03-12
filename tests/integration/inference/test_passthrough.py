# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Integration tests for the passthrough inference provider.

Spins up a lightweight mock OpenAI-compatible server and wires up
PassthroughInferenceAdapter against it, exercising the full path from
config validation through AsyncOpenAI client to a real HTTP endpoint.

Run with:
    uv run pytest tests/integration/inference/test_passthrough.py -x -q
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from llama_stack.providers.remote.inference.passthrough.config import PassthroughImplConfig
from llama_stack.providers.remote.inference.passthrough.passthrough import PassthroughInferenceAdapter

# -- minimal OpenAI-compatible mock server --


class _OpenAIHandler(BaseHTTPRequestHandler):
    """Records received headers and returns a minimal chat completion response."""

    received_headers: dict[str, str] = {}
    response_override: dict[str, Any] | None = None

    def do_GET(self):  # noqa: N802
        _OpenAIHandler.received_headers = dict(self.headers)
        if self.path == "/v1/models":
            payload = json.dumps({"data": [{"id": "mock-llm", "object": "model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_error(404)

    def do_POST(self):  # noqa: N802
        _OpenAIHandler.received_headers = dict(self.headers)
        length = int(self.headers.get("Content-Length", 0))
        _ = self.rfile.read(length)

        resp = self.response_override or {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "mock-llm",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        payload = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


@pytest.fixture(scope="module")
def mock_server():
    """Start a local OpenAI-compatible HTTP server."""
    server = HTTPServer(("127.0.0.1", 0), _OpenAIHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def _make_adapter(base_url: str, **kwargs) -> PassthroughInferenceAdapter:
    config = PassthroughImplConfig(base_url=base_url, **kwargs)  # type: ignore[arg-type]
    adapter = PassthroughInferenceAdapter(config)
    spec = MagicMock()
    spec.provider_data_validator = "llama_stack.providers.remote.inference.passthrough.PassthroughProviderDataValidator"
    spec.provider_type = "remote::passthrough"
    object.__setattr__(adapter, "__provider_spec__", spec)
    return adapter


# -- tests --


async def test_static_api_key_sent_as_bearer(mock_server):
    """Static api_key config is forwarded as Authorization: Bearer."""
    adapter = _make_adapter(mock_server, api_key=SecretStr("sk-static-key"))
    from llama_stack_api import OpenAIChatCompletionRequestWithExtraBody, OpenAIUserMessageParam

    request = OpenAIChatCompletionRequestWithExtraBody(
        model="mock-llm",
        messages=[OpenAIUserMessageParam(content="hi")],
    )
    await adapter.openai_chat_completion(request)
    assert _OpenAIHandler.received_headers.get("Authorization") == "Bearer sk-static-key"


async def test_forward_headers_sent_downstream(mock_server):
    """forward_headers config maps provider-data keys to outbound HTTP headers."""
    from llama_stack.core.request_headers import request_provider_data_context
    from llama_stack_api import OpenAIChatCompletionRequestWithExtraBody, OpenAIUserMessageParam

    adapter = _make_adapter(
        mock_server,
        forward_headers={"maas_token": "Authorization", "tenant_id": "X-Tenant-ID"},
    )

    provider_data = json.dumps({"maas_token": "Bearer maas-tok-abc", "tenant_id": "acme"})
    with request_provider_data_context({"x-llamastack-provider-data": provider_data}):
        request = OpenAIChatCompletionRequestWithExtraBody(
            model="mock-llm",
            messages=[OpenAIUserMessageParam(content="hi")],
        )
        await adapter.openai_chat_completion(request)

    assert _OpenAIHandler.received_headers.get("Authorization") == "Bearer maas-tok-abc"
    assert _OpenAIHandler.received_headers.get("X-Tenant-ID") == "acme"


async def test_default_deny_unlisted_keys_not_forwarded(mock_server):
    """Keys not in forward_headers are never sent downstream."""
    from llama_stack.core.request_headers import request_provider_data_context
    from llama_stack_api import OpenAIChatCompletionRequestWithExtraBody, OpenAIUserMessageParam

    adapter = _make_adapter(
        mock_server,
        forward_headers={"allowed_key": "X-Allowed"},
    )

    provider_data = json.dumps({"allowed_key": "allowed-value", "secret": "should-not-leak"})
    with request_provider_data_context({"x-llamastack-provider-data": provider_data}):
        request = OpenAIChatCompletionRequestWithExtraBody(
            model="mock-llm",
            messages=[OpenAIUserMessageParam(content="hi")],
        )
        await adapter.openai_chat_completion(request)

    headers_str = str(_OpenAIHandler.received_headers)
    assert "should-not-leak" not in headers_str
    assert _OpenAIHandler.received_headers.get("X-Allowed") == "allowed-value"


async def test_static_api_key_wins_over_forwarded_authorization(mock_server):
    """Static api_key takes precedence over a forwarded Authorization header."""
    from llama_stack.core.request_headers import request_provider_data_context
    from llama_stack_api import OpenAIChatCompletionRequestWithExtraBody, OpenAIUserMessageParam

    adapter = _make_adapter(
        mock_server,
        api_key=SecretStr("sk-static"),
        forward_headers={"user_token": "Authorization"},
    )

    provider_data = json.dumps({"user_token": "Bearer user-token"})
    with request_provider_data_context({"x-llamastack-provider-data": provider_data}):
        request = OpenAIChatCompletionRequestWithExtraBody(
            model="mock-llm",
            messages=[OpenAIUserMessageParam(content="hi")],
        )
        await adapter.openai_chat_completion(request)

    # only one Authorization header, with the static value
    auth = _OpenAIHandler.received_headers.get("Authorization")
    assert auth == "Bearer sk-static"


async def test_blocked_headers_rejected_at_config_time(mock_server):
    """Security-sensitive header names are rejected at config parse time."""
    from pydantic import ValidationError

    for blocked in ("Host", "Transfer-Encoding", "X-Forwarded-For", "Proxy-Authorization"):
        with pytest.raises(ValidationError, match="blocked"):
            _make_adapter(mock_server, forward_headers={"key": blocked})


async def test_extra_blocked_headers_enforced(mock_server):
    """extra_blocked_headers tightens the policy at config parse time."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="blocked"):
        _make_adapter(
            mock_server,
            forward_headers={"internal": "X-Internal-Debug"},
            extra_blocked_headers=["x-internal-debug"],
        )


async def test_no_forward_headers_no_crash(mock_server):
    """Provider works normally when forward_headers is not configured."""
    from llama_stack_api import OpenAIChatCompletionRequestWithExtraBody, OpenAIUserMessageParam

    adapter = _make_adapter(mock_server, api_key=SecretStr("sk-static"))
    request = OpenAIChatCompletionRequestWithExtraBody(
        model="mock-llm",
        messages=[OpenAIUserMessageParam(content="hello")],
    )
    response = await adapter.openai_chat_completion(request)
    assert response is not None

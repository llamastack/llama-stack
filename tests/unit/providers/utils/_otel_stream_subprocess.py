# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Subprocess script run under ``opentelemetry-instrument`` by test_otel_stream_resilience.py.

Streams an OpenAI chat completion containing ``arguments: null`` in a tool-call
delta through a mock HTTP transport.  When OTEL auto-instrumentation wraps the
stream and content capture is enabled, the unpatched ChoiceBuffer crashes with
TypeError during cleanup.

Set APPLY_OTEL_PATCH=1 to apply _patch_otel_choice_buffer() before streaming.
"""

import asyncio
import json
import os

import httpx
from openai import AsyncOpenAI

if os.environ.get("APPLY_OTEL_PATCH") == "1":
    from llama_stack.providers.utils.inference.stream_utils import _patch_otel_choice_buffer

    _patch_otel_choice_buffer()

CHUNKS = [
    {
        "id": "c",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "f", "arguments": None},
                        }
                    ],
                },
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "c",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]},
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "c",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
    },
]

SSE = "".join(f"data: {json.dumps(c)}\n\n" for c in CHUNKS) + "data: [DONE]\n\n"


def handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=SSE.encode())


async def main():
    client = AsyncOpenAI(
        api_key="fake",
        base_url="http://localhost:1/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    stream = await client.chat.completions.create(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    async for _ in stream:
        pass


asyncio.run(main())

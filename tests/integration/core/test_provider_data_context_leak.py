"""
Integration test for provider data context isolation in streaming requests.

This test verifies that PROVIDER_DATA_VAR doesn't leak between sequential
streaming requests, ensuring provider credentials and configuration are
properly isolated between requests.
"""

import json
import pytest


@pytest.mark.asyncio
async def test_provider_data_isolation_library_client():
    """
    Verifies that provider data context is properly isolated between
    sequential streaming requests and cleaned up after each request.
    """
    from llama_stack.core.request_headers import PROVIDER_DATA_VAR, request_provider_data_context
    from llama_stack.core.utils.context import preserve_contexts_async_generator

    async def mock_streaming_provider():
        """Simulates a streaming provider that reads PROVIDER_DATA_VAR"""
        provider_data = PROVIDER_DATA_VAR.get()
        yield {"provider_data": provider_data, "chunk": 1}

    async def sse_generator(gen):
        """Simulates the SSE generator in the server"""
        async for item in gen:
            yield f"data: {json.dumps(item)}\n\n"

    # Request 1: Set provider data to {"key": "value1"}
    headers1 = {"X-LlamaStack-Provider-Data": json.dumps({"key": "value1"})}
    with request_provider_data_context(headers1):
        gen1 = preserve_contexts_async_generator(
            sse_generator(mock_streaming_provider()),
            [PROVIDER_DATA_VAR]
        )

    chunks1 = [chunk async for chunk in gen1]
    data1 = json.loads(chunks1[0].split("data: ")[1])
    assert data1["provider_data"] == {"key": "value1"}

    # Context should be cleared after consuming the generator
    leaked_data = PROVIDER_DATA_VAR.get()
    assert leaked_data is None, f"Context leaked after request 1: {leaked_data}"

    # Request 2: Set different provider data {"key": "value2"}
    headers2 = {"X-LlamaStack-Provider-Data": json.dumps({"key": "value2"})}
    with request_provider_data_context(headers2):
        gen2 = preserve_contexts_async_generator(
            sse_generator(mock_streaming_provider()),
            [PROVIDER_DATA_VAR]
        )

    chunks2 = [chunk async for chunk in gen2]
    data2 = json.loads(chunks2[0].split("data: ")[1])
    assert data2["provider_data"] == {"key": "value2"}

    leaked_data2 = PROVIDER_DATA_VAR.get()
    assert leaked_data2 is None, f"Context leaked after request 2: {leaked_data2}"

    # Request 3: No provider data
    gen3 = preserve_contexts_async_generator(
        sse_generator(mock_streaming_provider()),
        [PROVIDER_DATA_VAR]
    )

    chunks3 = [chunk async for chunk in gen3]
    data3 = json.loads(chunks3[0].split("data: ")[1])
    assert data3["provider_data"] is None


@pytest.mark.skipif(
    True,
    reason="Requires custom test provider with context echo capability"
)
def test_provider_data_isolation_with_server(llama_stack_client):
    """
    Server-based test for context isolation (currently skipped).

    Requires a test inference provider that echoes back PROVIDER_DATA_VAR
    in streaming responses to verify proper isolation.
    """
    response1 = llama_stack_client.inference.chat_completion(
        model_id="context-echo-model",
        messages=[{"role": "user", "content": "test"}],
        stream=True,
        extra_headers={
            "X-LlamaStack-Provider-Data": json.dumps({"test_key": "value1"})
        },
    )

    chunks1 = []
    for chunk in response1:
        if chunk.choices and chunk.choices[0].delta.content:
            chunks1.append(chunk.choices[0].delta.content)

    response1_data = json.loads("".join(chunks1))
    assert response1_data["provider_data"] == {"test_key": "value1"}

    response2 = llama_stack_client.inference.chat_completion(
        model_id="context-echo-model",
        messages=[{"role": "user", "content": "test"}],
        stream=True,
        extra_headers={
            "X-LlamaStack-Provider-Data": json.dumps({"test_key": "value2"})
        },
    )

    chunks2 = []
    for chunk in response2:
        if chunk.choices and chunk.choices[0].delta.content:
            chunks2.append(chunk.choices[0].delta.content)

    response2_data = json.loads("".join(chunks2))
    assert response2_data["provider_data"] == {"test_key": "value2"}

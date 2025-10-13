# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock

import pytest

from llama_stack.apis.agents.openai_responses import (
    OpenAIResponseText,
)
from llama_stack.apis.safety import ModerationObject, ModerationObjectResults
from llama_stack.apis.tools import ToolDef
from llama_stack.providers.inline.agents.meta_reference.responses.streaming import (
    StreamingResponseOrchestrator,
    convert_tooldef_to_chat_tool,
)
from llama_stack.providers.inline.agents.meta_reference.responses.types import ChatCompletionContext


@pytest.fixture
def mock_safety_api():
    safety_api = AsyncMock()
    # Mock the routing table and shields list for guardrails lookup
    safety_api.routing_table = AsyncMock()
    shield = AsyncMock()
    shield.identifier = "llama-guard"
    shield.provider_resource_id = "llama-guard-model"
    safety_api.routing_table.list_shields.return_value = AsyncMock(data=[shield])
    # Mock run_moderation to return non-flagged result by default
    safety_api.run_moderation.return_value = AsyncMock(flagged=False)
    return safety_api


@pytest.fixture
def mock_inference_api():
    inference_api = AsyncMock()
    return inference_api


@pytest.fixture
def mock_context():
    context = AsyncMock(spec=ChatCompletionContext)
    # Add required attributes that StreamingResponseOrchestrator expects
    context.tool_context = AsyncMock()
    context.tool_context.previous_tools = {}
    context.messages = []
    return context


def test_convert_tooldef_to_chat_tool_preserves_items_field():
    """Test that array parameters preserve the items field during conversion.

    This test ensures that when converting ToolDef with array-type parameters
    to OpenAI ChatCompletionToolParam format, the 'items' field is preserved.
    Without this fix, array parameters would be missing schema information about their items.
    """
    tool_def = ToolDef(
        name="test_tool",
        description="A test tool with array parameter",
        input_schema={
            "type": "object",
            "properties": {"tags": {"type": "array", "description": "List of tags", "items": {"type": "string"}}},
            "required": ["tags"],
        },
    )

    result = convert_tooldef_to_chat_tool(tool_def)

    assert result["type"] == "function"
    assert result["function"]["name"] == "test_tool"

    tags_param = result["function"]["parameters"]["properties"]["tags"]
    assert tags_param["type"] == "array"
    assert "items" in tags_param, "items field should be preserved for array parameters"
    assert tags_param["items"] == {"type": "string"}


async def test_apply_guardrails_no_violation(mock_safety_api, mock_inference_api, mock_context):
    """Test guardrails validation with no violations."""
    text = "Hello world"
    guardrail_ids = ["llama-guard"]

    # Mock successful guardrails validation (no violation)
    mock_response = AsyncMock()
    mock_response.violation = None
    mock_safety_api.run_shield.return_value = mock_response

    # Create orchestrator with safety components
    orchestrator = StreamingResponseOrchestrator(
        inference_api=mock_inference_api,
        ctx=mock_context,
        response_id="test_id",
        created_at=1234567890,
        text=OpenAIResponseText(),
        max_infer_iters=5,
        tool_executor=AsyncMock(),
        safety_api=mock_safety_api,
        guardrail_ids=guardrail_ids,
    )

    result = await orchestrator._apply_guardrails(text)

    assert result is None
    # Verify run_moderation was called with the correct model
    mock_safety_api.run_moderation.assert_called_once()
    # Get the actual call arguments
    call_args = mock_safety_api.run_moderation.call_args
    assert call_args[1]["model"] == "llama-guard-model"  # The provider_resource_id from our mock


async def test_apply_guardrails_with_violation(mock_safety_api, mock_inference_api, mock_context):
    """Test guardrails validation with safety violation."""
    text = "Harmful content"
    guardrail_ids = ["llama-guard"]

    # Mock moderation to return flagged content
    flagged_result = ModerationObjectResults(flagged=True, categories={"violence": True})
    mock_moderation_object = ModerationObject(id="test-mod-id", model="llama-guard-model", results=[flagged_result])
    mock_safety_api.run_moderation.return_value = mock_moderation_object

    # Create orchestrator with safety components
    orchestrator = StreamingResponseOrchestrator(
        inference_api=mock_inference_api,
        ctx=mock_context,
        response_id="test_id",
        created_at=1234567890,
        text=OpenAIResponseText(),
        max_infer_iters=5,
        tool_executor=AsyncMock(),
        safety_api=mock_safety_api,
        guardrail_ids=guardrail_ids,
    )

    result = await orchestrator._apply_guardrails(text)

    assert result == "Content flagged by moderation"


async def test_apply_guardrails_empty_inputs(mock_safety_api, mock_inference_api, mock_context):
    """Test guardrails validation with empty inputs."""
    # Create orchestrator with safety components
    orchestrator = StreamingResponseOrchestrator(
        inference_api=mock_inference_api,
        ctx=mock_context,
        response_id="test_id",
        created_at=1234567890,
        text=OpenAIResponseText(),
        max_infer_iters=5,
        tool_executor=AsyncMock(),
        safety_api=mock_safety_api,
        guardrail_ids=[],
    )

    # Test empty guardrail_ids
    result = await orchestrator._apply_guardrails("test")
    assert result is None

    # Test empty text
    orchestrator.guardrail_ids = ["llama-guard"]
    result = await orchestrator._apply_guardrails("")
    assert result is None

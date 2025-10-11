# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock

import pytest

from llama_stack.apis.agents.openai_responses import (
    OpenAIResponseContentPartRefusal,
    OpenAIResponseText,
)
from llama_stack.apis.inference import UserMessage
from llama_stack.apis.safety import SafetyViolation, ViolationLevel
from llama_stack.apis.tools import ToolDef
from llama_stack.providers.inline.agents.meta_reference.responses.streaming import (
    StreamingResponseOrchestrator,
    convert_tooldef_to_chat_tool,
)
from llama_stack.providers.inline.agents.meta_reference.responses.types import ChatCompletionContext


@pytest.fixture
def mock_safety_api():
    safety_api = AsyncMock()
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


async def test_check_input_safety_no_violation(mock_safety_api, mock_inference_api, mock_context):
    """Test input shield validation with no violations."""
    messages = [UserMessage(content="Hello world")]
    shield_ids = ["llama-guard"]

    # Mock successful shield validation (no violation)
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
        shield_ids=shield_ids,
    )

    result = await orchestrator._check_input_safety(messages)

    assert result is None
    mock_safety_api.run_shield.assert_called_once_with(shield_id="llama-guard", messages=messages, params={})


async def test_check_input_safety_with_violation(mock_safety_api, mock_inference_api, mock_context):
    """Test input shield validation with safety violation."""
    messages = [UserMessage(content="Harmful content")]
    shield_ids = ["llama-guard"]

    # Mock shield violation
    violation = SafetyViolation(
        violation_level=ViolationLevel.ERROR, user_message="Content violates safety guidelines", metadata={}
    )
    mock_response = AsyncMock()
    mock_response.violation = violation
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
        shield_ids=shield_ids,
    )

    result = await orchestrator._check_input_safety(messages)

    assert isinstance(result, OpenAIResponseContentPartRefusal)
    assert result.refusal == "Content violates safety guidelines"


async def test_check_input_safety_empty_inputs(mock_safety_api, mock_inference_api, mock_context):
    """Test input shield validation with empty inputs."""
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
        shield_ids=[],
    )

    # Test empty shield_ids
    result = await orchestrator._check_input_safety([UserMessage(content="test")])
    assert result is None

    # Test empty messages
    orchestrator.shield_ids = ["llama-guard"]
    result = await orchestrator._check_input_safety([])
    assert result is None

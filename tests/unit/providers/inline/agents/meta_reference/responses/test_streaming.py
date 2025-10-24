# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock

import pytest

from llama_stack.apis.tools import ToolDef
from llama_stack.providers.inline.agents.meta_reference.responses.streaming import (
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

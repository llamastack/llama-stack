# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.tools import ToolDef
from llama_stack.providers.inline.agents.meta_reference.responses.streaming import (
    convert_tooldef_to_chat_tool,
)


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

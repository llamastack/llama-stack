# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Unit tests for MCP tool parameter conversion in streaming responses.

This tests the fix for handling array-type parameters with 'items' field
when converting MCP tool definitions to OpenAI format.
"""

from llama_stack.apis.tools import ToolDef, ToolParameter
from llama_stack.models.llama.datatypes import ToolDefinition, ToolParamDefinition
from llama_stack.providers.utils.inference.openai_compat import convert_tooldef_to_openai_tool


def test_mcp_tool_conversion_with_array_items():
    """
    Test that MCP tool parameters with array type and items field are properly converted.

    This is a regression test for the bug where array parameters without 'items'
    caused OpenAI API validation errors like:
    "Invalid schema for function 'pods_exec': In context=('properties', 'command'),
    array schema missing items."
    """
    # Create a tool parameter with array type and items specification
    # This mimics what kubernetes-mcp-server's pods_exec tool has
    tool_param = ToolParameter(
        name="command",
        parameter_type="array",
        description="Command to execute in the pod",
        required=True,
        items={"type": "string"},  # This is the crucial field
    )

    # Convert to ToolDefinition format (as done in streaming.py)
    tool_def = ToolDefinition(
        tool_name="test_tool",
        description="Test tool with array parameter",
        parameters={
            "command": ToolParamDefinition(
                param_type=tool_param.parameter_type,
                description=tool_param.description,
                required=tool_param.required,
                default=tool_param.default,
                items=tool_param.items,  # The fix: ensure items is passed through
            )
        },
    )

    # Convert to OpenAI format
    openai_tool = convert_tooldef_to_openai_tool(tool_def)

    # Verify the conversion includes the items field
    assert openai_tool["type"] == "function"
    assert openai_tool["function"]["name"] == "test_tool"
    assert "parameters" in openai_tool["function"]

    parameters = openai_tool["function"]["parameters"]
    assert "properties" in parameters
    assert "command" in parameters["properties"]

    command_param = parameters["properties"]["command"]
    assert command_param["type"] == "array"
    assert "items" in command_param, "Array parameter must have 'items' field for OpenAI API"
    assert command_param["items"] == {"type": "string"}


def test_mcp_tool_conversion_without_array():
    """Test that non-array parameters work correctly without items field."""
    tool_param = ToolParameter(
        name="name",
        parameter_type="string",
        description="Name parameter",
        required=True,
    )

    tool_def = ToolDefinition(
        tool_name="test_tool",
        description="Test tool with string parameter",
        parameters={
            "name": ToolParamDefinition(
                param_type=tool_param.parameter_type,
                description=tool_param.description,
                required=tool_param.required,
                items=tool_param.items,  # Will be None for non-array types
            )
        },
    )

    openai_tool = convert_tooldef_to_openai_tool(tool_def)

    # Verify basic structure
    assert openai_tool["type"] == "function"
    parameters = openai_tool["function"]["parameters"]
    assert "name" in parameters["properties"]

    name_param = parameters["properties"]["name"]
    assert name_param["type"] == "string"
    # items should not be present for non-array types
    assert "items" not in name_param or name_param.get("items") is None


def test_mcp_tool_conversion_complex_array_items():
    """Test array parameter with complex items schema (object type)."""
    tool_param = ToolParameter(
        name="configs",
        parameter_type="array",
        description="Array of configuration objects",
        required=False,
        items={
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["key"],
        },
    )

    tool_def = ToolDefinition(
        tool_name="test_tool",
        description="Test tool with complex array parameter",
        parameters={
            "configs": ToolParamDefinition(
                param_type=tool_param.parameter_type,
                description=tool_param.description,
                required=tool_param.required,
                items=tool_param.items,
            )
        },
    )

    openai_tool = convert_tooldef_to_openai_tool(tool_def)

    # Verify complex items schema is preserved
    parameters = openai_tool["function"]["parameters"]
    configs_param = parameters["properties"]["configs"]

    assert configs_param["type"] == "array"
    assert "items" in configs_param
    assert configs_param["items"]["type"] == "object"
    assert "properties" in configs_param["items"]
    assert "key" in configs_param["items"]["properties"]
    assert "value" in configs_param["items"]["properties"]
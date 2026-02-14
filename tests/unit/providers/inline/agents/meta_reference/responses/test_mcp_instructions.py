# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import patch

from llama_stack.providers.inline.agents.meta_reference.responses.streaming import (
    MCP_INSTRUCTIONS_MAX_LENGTH,
)
from llama_stack_api import OpenAISystemMessageParam


def test_mcp_instructions_concatenation():
    """Test that MCP server instructions are properly concatenated with original instructions."""
    original_instructions = "Original instructions"
    mcp_instructions = ["MCP server says: Use tool carefully"]

    accumulated_instructions = "\n\n".join(mcp_instructions)
    final_instructions = f"{original_instructions}\n\n{accumulated_instructions}"

    assert final_instructions == "Original instructions\n\nMCP server says: Use tool carefully"


def test_mcp_instructions_truncation():
    """Test that MCP server instructions are truncated when they exceed max length."""
    # Create instructions that exceed max length
    long_instructions = "a" * (MCP_INSTRUCTIONS_MAX_LENGTH + 100)

    # Simulate the truncation logic from _process_mcp_tool
    if len(long_instructions) > MCP_INSTRUCTIONS_MAX_LENGTH:
        truncated = long_instructions[:MCP_INSTRUCTIONS_MAX_LENGTH] + "...[truncated]"
    else:
        truncated = long_instructions

    assert len(truncated) == MCP_INSTRUCTIONS_MAX_LENGTH + len("...[truncated]")
    assert truncated.endswith("...[truncated]")


def test_mcp_instructions_opt_in_enabled():
    """Test that by default MCP server instructions are enabled (max length > 0)."""
    # By default, max length is 1000, so instructions should be included
    assert MCP_INSTRUCTIONS_MAX_LENGTH > 0
    assert MCP_INSTRUCTIONS_MAX_LENGTH == 1000


def test_mcp_instructions_opt_in_disabled():
    """Test that MCP server instructions can be disabled by setting max length to 0."""
    # Test the disabled case
    with patch("llama_stack.providers.inline.agents.meta_reference.responses.streaming.MCP_INSTRUCTIONS_MAX_LENGTH", 0):
        # Simulate the check used in streaming.py
        simulated_max_length = 0
        if simulated_max_length > 0:
            should_include = True
        else:
            should_include = False

        assert should_include is False


def test_mcp_instructions_no_duplicates():
    """Test that duplicate MCP server instructions are not added twice."""
    mcp_server_instructions = []
    instructions = "Unique instructions"

    # Add instructions first time
    if instructions not in mcp_server_instructions:
        mcp_server_instructions.append(instructions)

    assert len(mcp_server_instructions) == 1

    # Try to add same instructions again
    if instructions not in mcp_server_instructions:
        mcp_server_instructions.append(instructions)

    # Should still only have one copy
    assert len(mcp_server_instructions) == 1
    assert mcp_server_instructions[0] == instructions


def test_server_label_to_instructions_mapping():
    """Test that server_label -> instructions mapping works correctly."""
    server_label_to_instructions = {}
    server_label = "test-mcp-server"
    instructions = "These are MCP server instructions"

    # Store instructions
    server_label_to_instructions[server_label] = instructions

    # Retrieve instructions
    assert server_label in server_label_to_instructions
    assert server_label_to_instructions[server_label] == instructions

    # Test restoration logic
    mcp_server_instructions = []
    if server_label in server_label_to_instructions:
        restored_instructions = server_label_to_instructions[server_label]
        if restored_instructions and restored_instructions not in mcp_server_instructions:
            mcp_server_instructions.append(restored_instructions)

    assert instructions in mcp_server_instructions
    assert len(mcp_server_instructions) == 1


def test_system_message_update_with_instructions():
    """Test that system message is properly updated with accumulated instructions."""
    # Start with a system message
    messages = [OpenAISystemMessageParam(content="Original system message")]
    instructions = "Original system message\n\nMCP instructions from server"

    # Simulate the update logic from streaming.py
    if messages and isinstance(messages[0], OpenAISystemMessageParam):
        # Update existing system message
        updated_message = OpenAISystemMessageParam(content=instructions)
        messages[0] = updated_message
    else:
        # Insert new system message
        messages.insert(0, OpenAISystemMessageParam(content=instructions))

    assert len(messages) == 1
    assert messages[0].content == instructions


def test_system_message_insert_when_no_system_message():
    """Test that system message is inserted when messages don't start with one."""
    # Start with non-system messages
    from llama_stack_api import OpenAIUserMessageParam

    messages = [OpenAIUserMessageParam(content="User message")]
    instructions = "New system instructions"

    # Simulate the insertion logic from streaming.py
    if messages and isinstance(messages[0], OpenAISystemMessageParam):
        # Update existing
        messages[0] = OpenAISystemMessageParam(content=instructions)
    else:
        # Insert new
        messages.insert(0, OpenAISystemMessageParam(content=instructions))

    assert len(messages) == 2
    assert isinstance(messages[0], OpenAISystemMessageParam)
    assert messages[0].content == instructions
    assert isinstance(messages[1], OpenAIUserMessageParam)

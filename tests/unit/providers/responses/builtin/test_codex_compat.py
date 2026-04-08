# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for Codex CLI compatibility.

External clients like OpenAI Codex send tool types (``local_shell``,
``tool_search``, ``custom``) and input item types (``local_shell_call``,
``custom_tool_call``) that are not part of the standard OpenAI Responses API.
These tests verify that Llama Stack accepts and handles them gracefully.
"""

from unittest.mock import AsyncMock

import pytest

from llama_stack.providers.inline.responses.builtin.responses.types import ToolContext
from llama_stack.providers.inline.responses.builtin.responses.utils import (
    convert_response_input_to_chat_messages,
)
from llama_stack_api.openai_responses import (
    OpenAIResponseInputToolCustom,
    OpenAIResponseInputToolFileSearch,
    OpenAIResponseInputToolFunction,
    OpenAIResponseInputToolMCP,
    OpenAIResponseInputToolWebSearch,
    OpenAIResponseInputUnknown,
    OpenAIResponseMessage,
)

# ---------------------------------------------------------------------------
# Model-level parsing tests (Pydantic validation)
# ---------------------------------------------------------------------------


class TestOpenAIResponseInputToolCustomParsing:
    """Verify that the catch-all tool type parses various tool definitions."""

    def test_custom_tool_type(self):
        tool = OpenAIResponseInputToolCustom.model_validate(
            {"type": "custom", "name": "apply_patch", "description": "Apply a patch"}
        )
        assert tool.type == "custom"
        assert tool.name == "apply_patch"

    def test_local_shell_tool_type(self):
        tool = OpenAIResponseInputToolCustom.model_validate({"type": "local_shell"})
        assert tool.type == "local_shell"
        assert tool.name is None

    def test_tool_search_tool_type(self):
        tool = OpenAIResponseInputToolCustom.model_validate(
            {"type": "tool_search", "execution": "client", "description": "Search", "parameters": {}}
        )
        assert tool.type == "tool_search"

    def test_image_generation_tool_type(self):
        tool = OpenAIResponseInputToolCustom.model_validate({"type": "image_generation", "output_format": "png"})
        assert tool.type == "image_generation"

    def test_custom_tool_with_format_field(self):
        """Codex sends freeform tools with a ``format`` field."""
        tool = OpenAIResponseInputToolCustom.model_validate(
            {
                "type": "custom",
                "name": "apply_patch",
                "description": "Apply a patch",
                "format": {"type": "json_schema", "syntax": "diff", "definition": "patch format"},
            }
        )
        assert tool.type == "custom"
        assert tool.name == "apply_patch"
        # Extra fields are preserved
        assert tool.model_extra is not None
        assert tool.model_extra["format"]["syntax"] == "diff"


class TestOpenAIResponseInputUnknownParsing:
    """Verify that the catch-all input item type parses various items."""

    def test_local_shell_call(self):
        item = OpenAIResponseInputUnknown.model_validate(
            {
                "type": "local_shell_call",
                "call_id": "call_abc",
                "status": "completed",
                "action": {"type": "exec", "command": ["ls", "src"]},
            }
        )
        assert item.type == "local_shell_call"
        assert item.model_extra is not None
        assert item.model_extra["call_id"] == "call_abc"

    def test_local_shell_call_output(self):
        item = OpenAIResponseInputUnknown.model_validate(
            {
                "type": "local_shell_call_output",
                "call_id": "call_abc",
                "output": "file1.py\nfile2.py",
            }
        )
        assert item.type == "local_shell_call_output"

    def test_custom_tool_call(self):
        item = OpenAIResponseInputUnknown.model_validate(
            {
                "type": "custom_tool_call",
                "call_id": "call_def",
                "name": "apply_patch",
                "input": "some data",
                "status": "completed",
            }
        )
        assert item.type == "custom_tool_call"

    def test_custom_tool_call_output(self):
        item = OpenAIResponseInputUnknown.model_validate(
            {
                "type": "custom_tool_call_output",
                "call_id": "call_def",
                "output": "patch applied",
            }
        )
        assert item.type == "custom_tool_call_output"

    def test_tool_search_call(self):
        item = OpenAIResponseInputUnknown.model_validate(
            {
                "type": "tool_search_call",
                "call_id": "call_ts",
                "execution": "client",
                "arguments": {},
            }
        )
        assert item.type == "tool_search_call"

    def test_ghost_snapshot(self):
        item = OpenAIResponseInputUnknown.model_validate(
            {
                "type": "ghost_snapshot",
                "ghost_commit": {"sha": "abc123"},
            }
        )
        assert item.type == "ghost_snapshot"


# ---------------------------------------------------------------------------
# Union discrimination tests
# ---------------------------------------------------------------------------


class TestToolUnionDiscrimination:
    """Verify that known tool types resolve to specific models and unknown
    types fall through to ``OpenAIResponseInputToolCustom``."""

    def _parse_tool(self, data: dict):
        """Parse a tool dict through the OpenAIResponseInputTool union."""
        from pydantic import TypeAdapter

        from llama_stack_api.openai_responses import OpenAIResponseInputTool

        adapter = TypeAdapter(OpenAIResponseInputTool)
        return adapter.validate_python(data)

    def test_function_tool_resolves_correctly(self):
        tool = self._parse_tool({"type": "function", "name": "shell", "parameters": {"type": "object"}})
        assert isinstance(tool, OpenAIResponseInputToolFunction)

    def test_web_search_tool_resolves_correctly(self):
        tool = self._parse_tool({"type": "web_search"})
        assert isinstance(tool, OpenAIResponseInputToolWebSearch)

    def test_file_search_tool_resolves_correctly(self):
        tool = self._parse_tool({"type": "file_search", "vector_store_ids": ["vs_1"]})
        assert isinstance(tool, OpenAIResponseInputToolFileSearch)

    def test_mcp_tool_resolves_correctly(self):
        tool = self._parse_tool({"type": "mcp", "server_label": "my_server", "server_url": "http://localhost"})
        assert isinstance(tool, OpenAIResponseInputToolMCP)

    def test_local_shell_falls_through_to_custom(self):
        tool = self._parse_tool({"type": "local_shell"})
        assert isinstance(tool, OpenAIResponseInputToolCustom)
        assert tool.type == "local_shell"

    def test_custom_tool_resolves_to_custom(self):
        tool = self._parse_tool({"type": "custom", "name": "freeform"})
        assert isinstance(tool, OpenAIResponseInputToolCustom)
        assert tool.type == "custom"

    def test_tool_search_falls_through_to_custom(self):
        tool = self._parse_tool({"type": "tool_search", "execution": "client", "description": "s", "parameters": {}})
        assert isinstance(tool, OpenAIResponseInputToolCustom)

    def test_image_generation_falls_through_to_custom(self):
        tool = self._parse_tool({"type": "image_generation", "output_format": "png"})
        assert isinstance(tool, OpenAIResponseInputToolCustom)


class TestInputUnionDiscrimination:
    """Verify that known input item types resolve correctly and unknown
    types fall through to ``OpenAIResponseInputUnknown``."""

    def _parse_input(self, data: dict):
        from pydantic import TypeAdapter

        from llama_stack_api.openai_responses import OpenAIResponseInput

        adapter = TypeAdapter(OpenAIResponseInput)
        return adapter.validate_python(data)

    def test_user_message_resolves_correctly(self):
        item = self._parse_input({"role": "user", "content": "hello"})
        assert isinstance(item, OpenAIResponseMessage)

    def test_local_shell_call_falls_through_to_unknown(self):
        item = self._parse_input({"type": "local_shell_call", "call_id": "c1", "status": "completed", "action": {}})
        assert isinstance(item, OpenAIResponseInputUnknown)
        assert item.type == "local_shell_call"

    def test_custom_tool_call_falls_through_to_unknown(self):
        item = self._parse_input({"type": "custom_tool_call", "call_id": "c2", "name": "apply_patch", "input": "data"})
        assert isinstance(item, OpenAIResponseInputUnknown)

    def test_ghost_snapshot_falls_through_to_unknown(self):
        item = self._parse_input({"type": "ghost_snapshot", "ghost_commit": {}})
        assert isinstance(item, OpenAIResponseInputUnknown)


# ---------------------------------------------------------------------------
# ToolContext tests
# ---------------------------------------------------------------------------


class TestToolContextWithCustomTools:
    """Verify ``ToolContext.available_tools`` handles custom/unknown tool types."""

    def test_custom_tool_passes_through(self):
        tools = [
            OpenAIResponseInputToolFunction(name="shell", parameters={"type": "object"}),
            OpenAIResponseInputToolCustom(type="local_shell"),
            OpenAIResponseInputToolCustom(type="custom", name="apply_patch"),
        ]
        ctx = ToolContext(tools)
        available = ctx.available_tools()
        assert len(available) == 3
        # The custom tools are returned as-is
        assert available[1].type == "local_shell"
        assert available[2].type == "custom"


# ---------------------------------------------------------------------------
# Chat conversion tests
# ---------------------------------------------------------------------------


class TestChatConversionWithUnknownInputItems:
    """Verify ``convert_response_input_to_chat_messages`` skips unknown items."""

    @pytest.fixture
    def mock_files_api(self):
        return AsyncMock()

    async def test_unknown_items_are_skipped(self, mock_files_api):
        """Unknown input items should be silently skipped, not cause errors."""
        input_items = [
            OpenAIResponseMessage(role="user", content="Hello"),
            OpenAIResponseInputUnknown(type="local_shell_call", call_id="c1"),
            OpenAIResponseInputUnknown(type="local_shell_call_output", call_id="c1"),
            OpenAIResponseMessage(role="user", content="What happened?"),
        ]
        messages = await convert_response_input_to_chat_messages(input_items, mock_files_api)
        # Only the two user messages should be converted
        assert len(messages) == 2
        assert messages[0].content == "Hello"
        assert messages[1].content == "What happened?"

    async def test_mixed_known_and_unknown_items(self, mock_files_api):
        """A mix of message, function_call, and unknown items should work."""
        from llama_stack_api.openai_responses import (
            OpenAIResponseInputFunctionToolCallOutput,
            OpenAIResponseOutputMessageFunctionToolCall,
        )

        input_items = [
            OpenAIResponseMessage(role="user", content="Run ls"),
            OpenAIResponseOutputMessageFunctionToolCall(
                call_id="fc_1",
                name="shell",
                arguments='{"command": "ls"}',
            ),
            OpenAIResponseInputFunctionToolCallOutput(
                call_id="fc_1",
                output="file1.py\nfile2.py",
            ),
            # These Codex-specific items should be skipped
            OpenAIResponseInputUnknown(type="local_shell_call", call_id="ls_1"),
            OpenAIResponseInputUnknown(type="local_shell_call_output", call_id="ls_1"),
            OpenAIResponseMessage(role="user", content="Good"),
        ]
        messages = await convert_response_input_to_chat_messages(input_items, mock_files_api)
        # user("Run ls") + assistant(tool_call) + tool(output) + user("Good")
        assert len(messages) == 4

    async def test_only_unknown_items(self, mock_files_api):
        """Input with only unknown items should produce no messages."""
        input_items = [
            OpenAIResponseInputUnknown(type="ghost_snapshot", ghost_commit={}),
            OpenAIResponseInputUnknown(type="custom_tool_call", call_id="c1"),
        ]
        messages = await convert_response_input_to_chat_messages(input_items, mock_files_api)
        assert len(messages) == 0


# ---------------------------------------------------------------------------
# OpenAIResponseText verbosity field
# ---------------------------------------------------------------------------


class TestResponseTextVerbosity:
    """Verify the ``verbosity`` field on ``OpenAIResponseText``."""

    def test_verbosity_low(self):
        from llama_stack_api.openai_responses import OpenAIResponseText

        text = OpenAIResponseText(verbosity="low")
        assert text.verbosity == "low"

    def test_verbosity_none_by_default(self):
        from llama_stack_api.openai_responses import OpenAIResponseText

        text = OpenAIResponseText()
        assert text.verbosity is None

    def test_verbosity_with_format(self):
        from llama_stack_api.openai_responses import OpenAIResponseText

        text = OpenAIResponseText(
            verbosity="high",
            format={"type": "text"},
        )
        assert text.verbosity == "high"
        assert text.format is not None

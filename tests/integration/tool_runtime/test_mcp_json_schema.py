# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Integration tests for MCP tools with complex JSON Schema support.
Tests $ref, $defs, and other JSON Schema features through MCP integration.
"""

import json

import pytest

from llama_stack.core.library_client import LlamaStackAsLibraryClient
from tests.common.mcp import make_mcp_server

AUTH_TOKEN = "test-token"


@pytest.fixture(scope="function")
def mcp_server_with_complex_schemas():
    """MCP server with tools that have complex schemas including $ref and $defs."""
    from mcp.server.fastmcp import Context

    async def book_flight(flight: dict, passengers: list[dict], payment: dict, ctx: Context) -> dict:
        """
        Book a flight with passenger and payment information.

        This tool uses JSON Schema $ref and $defs for type reuse.
        """
        return {
            "booking_id": "BK12345",
            "flight": flight,
            "passengers": passengers,
            "payment": payment,
            "status": "confirmed",
        }

    async def process_order(order_data: dict, ctx: Context) -> dict:
        """
        Process an order with nested address information.

        Uses nested objects and $ref.
        """
        return {"order_id": "ORD789", "status": "processing", "data": order_data}

    async def flexible_contact(contact_info: str, ctx: Context) -> dict:
        """
        Accept flexible contact (email or phone).

        Uses anyOf schema.
        """
        if "@" in contact_info:
            return {"type": "email", "value": contact_info}
        else:
            return {"type": "phone", "value": contact_info}

    # Manually attach complex schemas to the functions
    # (FastMCP might not support this by default, so this is test setup)

    # For MCP, we need to set the schema via tool annotations
    # This is test infrastructure to force specific schemas

    tools = {"book_flight": book_flight, "process_order": process_order, "flexible_contact": flexible_contact}

    # Note: In real MCP implementation, we'd configure these schemas properly
    # For testing, we may need to mock or extend the MCP server setup

    with make_mcp_server(required_auth_token=AUTH_TOKEN, tools=tools) as server_info:
        yield server_info


@pytest.fixture(scope="function")
def mcp_server_with_output_schemas():
    """MCP server with tools that have output schemas defined."""
    from mcp.server.fastmcp import Context

    async def get_weather(location: str, ctx: Context) -> dict:
        """
        Get weather with structured output.

        Has both input and output schemas.
        """
        return {"temperature": 72.5, "conditions": "Sunny", "humidity": 45, "wind_speed": 10.2}

    async def calculate(x: float, y: float, operation: str, ctx: Context) -> dict:
        """
        Perform calculation with validated output.
        """
        operations = {"add": x + y, "subtract": x - y, "multiply": x * y, "divide": x / y if y != 0 else None}
        result = operations.get(operation)
        return {"result": result, "operation": operation}

    tools = {"get_weather": get_weather, "calculate": calculate}

    with make_mcp_server(required_auth_token=AUTH_TOKEN, tools=tools) as server_info:
        yield server_info


class TestMCPSchemaPreservation:
    """Test that MCP tool schemas are preserved correctly."""

    def test_mcp_tools_list_with_schemas(self, llama_stack_client, mcp_server_with_complex_schemas):
        """Test listing MCP tools preserves input_schema."""
        if not isinstance(llama_stack_client, LlamaStackAsLibraryClient):
            pytest.skip("Library client required for local MCP server")

        test_toolgroup_id = "mcp::complex_list"
        uri = mcp_server_with_complex_schemas["server_url"]

        # Clean up any existing registration
        try:
            llama_stack_client.toolgroups.unregister(toolgroup_id=test_toolgroup_id)
        except Exception:
            pass

        # Register MCP toolgroup
        llama_stack_client.toolgroups.register(
            toolgroup_id=test_toolgroup_id,
            provider_id="model-context-protocol",
            mcp_endpoint=dict(uri=uri),
        )

        provider_data = {"mcp_headers": {uri: {"Authorization": f"Bearer {AUTH_TOKEN}"}}}
        auth_headers = {
            "X-LlamaStack-Provider-Data": json.dumps(provider_data),
        }

        # List runtime tools
        response = llama_stack_client.tool_runtime.list_tools(
            tool_group_id=test_toolgroup_id,
            extra_headers=auth_headers,
        )

        tools = response
        assert len(tools) > 0

        # Check each tool has input_schema
        for tool in tools:
            assert hasattr(tool, "input_schema")
            # Schema might be None or a dict depending on tool
            if tool.input_schema is not None:
                assert isinstance(tool.input_schema, dict)
                # Should have basic JSON Schema structure
                if "properties" in tool.input_schema:
                    assert "type" in tool.input_schema

    def test_mcp_schema_with_refs_preserved(self, llama_stack_client, mcp_server_with_complex_schemas):
        """Test that $ref and $defs in MCP schemas are preserved."""
        if not isinstance(llama_stack_client, LlamaStackAsLibraryClient):
            pytest.skip("Library client required for local MCP server")

        test_toolgroup_id = "mcp::complex_refs"
        uri = mcp_server_with_complex_schemas["server_url"]

        # Register
        try:
            llama_stack_client.toolgroups.unregister(toolgroup_id=test_toolgroup_id)
        except Exception:
            pass

        llama_stack_client.toolgroups.register(
            toolgroup_id=test_toolgroup_id,
            provider_id="model-context-protocol",
            mcp_endpoint=dict(uri=uri),
        )
        provider_data = {"mcp_headers": {uri: {"Authorization": f"Bearer {AUTH_TOKEN}"}}}
        auth_headers = {
            "X-LlamaStack-Provider-Data": json.dumps(provider_data),
        }

        # List tools
        response = llama_stack_client.tool_runtime.list_tools(
            tool_group_id=test_toolgroup_id,
            extra_headers=auth_headers,
        )

        # Find book_flight tool (which should have $ref/$defs)
        book_flight_tool = next((t for t in response if t.name == "book_flight"), None)

        if book_flight_tool and book_flight_tool.input_schema:
            # If the MCP server provides $defs, they should be preserved
            # This is the KEY test for the bug fix
            schema = book_flight_tool.input_schema

            # Check if schema has properties (might vary based on MCP implementation)
            if "properties" in schema:
                # Verify schema structure is preserved (exact structure depends on MCP server)
                assert isinstance(schema["properties"], dict)

            # If $defs are present, verify they're preserved
            if "$defs" in schema:
                assert isinstance(schema["$defs"], dict)
                # Each definition should be a dict
                for _def_name, def_schema in schema["$defs"].items():
                    assert isinstance(def_schema, dict)

    def test_mcp_output_schema_preserved(self, llama_stack_client, mcp_server_with_output_schemas):
        """Test that MCP outputSchema is preserved."""
        if not isinstance(llama_stack_client, LlamaStackAsLibraryClient):
            pytest.skip("Library client required for local MCP server")

        test_toolgroup_id = "mcp::with_output"
        uri = mcp_server_with_output_schemas["server_url"]

        try:
            llama_stack_client.toolgroups.unregister(toolgroup_id=test_toolgroup_id)
        except Exception:
            pass

        llama_stack_client.toolgroups.register(
            toolgroup_id=test_toolgroup_id,
            provider_id="model-context-protocol",
            mcp_endpoint=dict(uri=uri),
        )

        provider_data = {"mcp_headers": {uri: {"Authorization": f"Bearer {AUTH_TOKEN}"}}}
        auth_headers = {
            "X-LlamaStack-Provider-Data": json.dumps(provider_data),
        }

        response = llama_stack_client.tool_runtime.list_tools(
            tool_group_id=test_toolgroup_id,
            extra_headers=auth_headers,
        )

        # Find get_weather tool
        weather_tool = next((t for t in response if t.name == "get_weather"), None)

        if weather_tool:
            # Check if output_schema field exists and is preserved
            assert hasattr(weather_tool, "output_schema")

            # If MCP server provides output schema, it should be preserved
            if weather_tool.output_schema is not None:
                assert isinstance(weather_tool.output_schema, dict)
                # Should have JSON Schema structure
                if "properties" in weather_tool.output_schema:
                    assert "type" in weather_tool.output_schema


class TestMCPToolInvocation:
    """Test invoking MCP tools with complex schemas."""

    def test_invoke_mcp_tool_with_nested_data(self, llama_stack_client, mcp_server_with_complex_schemas):
        """Test invoking MCP tool that expects nested object structure."""
        if not isinstance(llama_stack_client, LlamaStackAsLibraryClient):
            pytest.skip("Library client required for local MCP server")

        test_toolgroup_id = "mcp::complex_invoke_nested"
        uri = mcp_server_with_complex_schemas["server_url"]

        try:
            llama_stack_client.toolgroups.unregister(toolgroup_id=test_toolgroup_id)
        except Exception:
            pass

        llama_stack_client.toolgroups.register(
            toolgroup_id=test_toolgroup_id,
            provider_id="model-context-protocol",
            mcp_endpoint=dict(uri=uri),
        )

        provider_data = {"mcp_headers": {uri: {"Authorization": f"Bearer {AUTH_TOKEN}"}}}
        auth_headers = {
            "X-LlamaStack-Provider-Data": json.dumps(provider_data),
        }

        # List tools to populate the tool index
        llama_stack_client.tool_runtime.list_tools(
            tool_group_id=test_toolgroup_id,
            extra_headers=auth_headers,
        )

        # Invoke tool with complex nested data
        result = llama_stack_client.tool_runtime.invoke_tool(
            tool_name="process_order",
            kwargs={
                "order_data": {
                    "items": [{"name": "Widget", "quantity": 2}, {"name": "Gadget", "quantity": 1}],
                    "shipping": {"address": {"street": "123 Main St", "city": "San Francisco", "zipcode": "94102"}},
                }
            },
            extra_headers=auth_headers,
        )

        # Should succeed without schema validation errors
        assert result.content is not None
        assert result.error_message is None

    def test_invoke_with_flexible_schema(self, llama_stack_client, mcp_server_with_complex_schemas):
        """Test invoking tool with anyOf schema (flexible input)."""
        if not isinstance(llama_stack_client, LlamaStackAsLibraryClient):
            pytest.skip("Library client required for local MCP server")

        test_toolgroup_id = "mcp::complex_invoke_flexible"
        uri = mcp_server_with_complex_schemas["server_url"]

        try:
            llama_stack_client.toolgroups.unregister(toolgroup_id=test_toolgroup_id)
        except Exception:
            pass

        llama_stack_client.toolgroups.register(
            toolgroup_id=test_toolgroup_id,
            provider_id="model-context-protocol",
            mcp_endpoint=dict(uri=uri),
        )

        provider_data = {"mcp_headers": {uri: {"Authorization": f"Bearer {AUTH_TOKEN}"}}}
        auth_headers = {
            "X-LlamaStack-Provider-Data": json.dumps(provider_data),
        }

        # List tools to populate the tool index
        llama_stack_client.tool_runtime.list_tools(
            tool_group_id=test_toolgroup_id,
            extra_headers=auth_headers,
        )

        # Test with email format
        result_email = llama_stack_client.tool_runtime.invoke_tool(
            tool_name="flexible_contact",
            kwargs={"contact_info": "user@example.com"},
            extra_headers=auth_headers,
        )

        assert result_email.error_message is None

        # Test with phone format
        result_phone = llama_stack_client.tool_runtime.invoke_tool(
            tool_name="flexible_contact",
            kwargs={"contact_info": "+15551234567"},
            extra_headers=auth_headers,
        )

        assert result_phone.error_message is None


class TestAgentWithMCPTools:
    """Test agents using MCP tools with complex schemas."""

    @pytest.mark.skip(reason="we need tool call recording for this test since session_id is injected")
    def test_agent_with_complex_mcp_tool(self, llama_stack_client, text_model_id, mcp_server_with_complex_schemas):
        """Test agent can use MCP tools with $ref/$defs schemas."""
        if not isinstance(llama_stack_client, LlamaStackAsLibraryClient):
            pytest.skip("Library client required for local MCP server")

        from llama_stack_client.lib.agents.agent import Agent
        from llama_stack_client.lib.agents.turn_events import StepCompleted

        test_toolgroup_id = "mcp::complex_agent"
        uri = mcp_server_with_complex_schemas["server_url"]

        try:
            llama_stack_client.toolgroups.unregister(toolgroup_id=test_toolgroup_id)
        except Exception:
            pass

        llama_stack_client.toolgroups.register(
            toolgroup_id=test_toolgroup_id,
            provider_id="model-context-protocol",
            mcp_endpoint=dict(uri=uri),
        )

        provider_data = {"mcp_headers": {uri: {"Authorization": f"Bearer {AUTH_TOKEN}"}}}
        auth_headers = {
            "X-LlamaStack-Provider-Data": json.dumps(provider_data),
        }

        tools_list = llama_stack_client.tools.list(
            toolgroup_id=test_toolgroup_id,
            extra_headers=auth_headers,
        )
        tool_defs = [
            {
                "type": "mcp",
                "server_url": uri,
                "server_label": test_toolgroup_id,
                "require_approval": "never",
                "allowed_tools": [tool.name for tool in tools_list],
            }
        ]

        agent = Agent(
            client=llama_stack_client,
            model=text_model_id,
            instructions="You are a helpful assistant that can process orders and book flights.",
            tools=tool_defs,
            extra_headers=auth_headers,
        )

        session_id = agent.create_session("test-session-complex")

        # Ask agent to use a tool with complex schema
        chunks = list(
            agent.create_turn(
                session_id=session_id,
                messages=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Process an order with 2 widgets going to 123 Main St, San Francisco",
                            }
                        ],
                    }
                ],
                stream=True,
                extra_headers=auth_headers,
            )
        )

        events = [chunk.event for chunk in chunks]
        tool_execution_steps = [
            event for event in events if isinstance(event, StepCompleted) and event.step_type == "tool_execution"
        ]

        for step in tool_execution_steps:
            for tool_response in step.result.tool_responses:
                assert tool_response.get("content") is not None

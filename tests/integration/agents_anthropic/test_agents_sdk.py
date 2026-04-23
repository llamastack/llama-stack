# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Integration tests for the Agents API using the Anthropic SDK.

These tests verify that the Agents API is compatible with the official
Anthropic Python SDK, ensuring real Claude Agent SDK code will work.
"""

import pytest

anthropic = pytest.importorskip("anthropic", reason="anthropic package not installed")


@pytest.fixture
def anthropic_client(agents_base_url):
    """Provide an Anthropic client pointed at our Llama Stack server."""
    # The Anthropic SDK expects base_url without the /v1alpha suffix
    # It will append /v1/agents internally
    return anthropic.Anthropic(
        base_url=agents_base_url,
        api_key="fake-key",  # Not validated by Llama Stack
    )


def test_sdk_create_agent(anthropic_client, text_model_id):
    """Create an agent using the Anthropic SDK."""
    try:
        agent = anthropic_client.beta.agents.create(
            model=text_model_id,
            name="SDK Test Agent",
            system="You are a helpful assistant.",
        )
    except AttributeError:
        pytest.skip("Anthropic SDK does not support beta.agents API yet")

    assert agent.id.startswith("agent_")
    assert agent.model == text_model_id
    assert agent.name == "SDK Test Agent"
    assert agent.version == 1


def test_sdk_create_agent_with_tools(anthropic_client, text_model_id):
    """Create an agent with tools using the SDK."""
    try:
        agent = anthropic_client.beta.agents.create(
            model=text_model_id,
            name="Tool Agent",
            system="You are a helpful assistant.",
            tools=[
                {
                    "type": "agent_toolset_20260401",
                    "configs": [
                        {"name": "web_search", "enabled": True},
                    ],
                },
                {
                    "type": "custom",
                    "name": "calculate",
                    "description": "Perform a calculation",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"},
                        },
                        "required": ["expression"],
                    },
                },
            ],
        )
    except AttributeError:
        pytest.skip("Anthropic SDK does not support beta.agents API yet")

    assert len(agent.tools) == 2
    assert agent.tools[0]["type"] == "agent_toolset_20260401"
    assert agent.tools[1]["type"] == "custom"
    assert agent.tools[1]["name"] == "calculate"


def test_sdk_list_agents(anthropic_client, text_model_id):
    """List agents using the SDK."""
    try:
        # Create an agent first
        created_agent = anthropic_client.beta.agents.create(
            model=text_model_id,
            name="List Test",
            system="Test",
        )

        # List agents
        agents_list = anthropic_client.beta.agents.list()
    except AttributeError:
        pytest.skip("Anthropic SDK does not support beta.agents API yet")

    assert hasattr(agents_list, "data")
    agent_ids = [a.id for a in agents_list.data]
    assert created_agent.id in agent_ids


def test_sdk_get_agent(anthropic_client, text_model_id):
    """Retrieve an agent by ID using the SDK."""
    try:
        # Create an agent
        created = anthropic_client.beta.agents.create(
            model=text_model_id,
            name="Get Test",
            system="Test agent",
        )

        # Retrieve it
        retrieved = anthropic_client.beta.agents.retrieve(created.id)
    except AttributeError:
        pytest.skip("Anthropic SDK does not support beta.agents API yet")

    assert retrieved.id == created.id
    assert retrieved.system == "Test agent"
    assert retrieved.name == "Get Test"


def test_sdk_update_agent(anthropic_client, text_model_id):
    """Update an agent using the SDK."""
    try:
        # Create an agent
        created = anthropic_client.beta.agents.create(
            model=text_model_id,
            name="Original",
            system="Original",
        )

        # Update it
        updated = anthropic_client.beta.agents.update(
            created.id,
            name="Updated",
            system="Updated",
        )
    except AttributeError:
        pytest.skip("Anthropic SDK does not support beta.agents API yet")

    assert updated.id == created.id
    assert updated.version == 2
    assert updated.name == "Updated"
    assert updated.system == "Updated"


def test_sdk_archive_agent(anthropic_client, text_model_id):
    """Archive an agent using the SDK."""
    try:
        # Create an agent
        created = anthropic_client.beta.agents.create(
            model=text_model_id,
            name="Archive Test",
        )

        # Archive it
        archived = anthropic_client.beta.agents.archive(created.id)
    except AttributeError:
        pytest.skip("Anthropic SDK does not support beta.agents API yet")

    assert archived.id == created.id
    assert archived.archived_at is not None

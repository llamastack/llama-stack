# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Integration tests for the Anthropic Agents API (/v1alpha/agents).

These tests verify agent configuration CRUD operations.
"""

import pytest


def test_create_agent_basic(agents_client, text_model_id):
    """Create a basic agent with model, name, and system prompt."""
    response = agents_client.post(
        "/agents",
        json={
            "model": text_model_id,
            "name": "Test Agent",
            "system": "You are a helpful coding assistant.",
        },
    )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    agent = response.json()
    assert agent["type"] == "agent"
    assert agent["id"].startswith("agent_")
    assert agent["model"] == text_model_id
    assert agent["name"] == "Test Agent"
    assert agent["system"] == "You are a helpful coding assistant."
    assert agent["version"] == 1
    assert agent["archived_at"] is None
    assert "created_at" in agent
    assert "updated_at" in agent


def test_create_agent_with_tools(agents_client, text_model_id):
    """Create an agent with tool definitions."""
    response = agents_client.post(
        "/agents",
        json={
            "model": text_model_id,
            "name": "Tool Agent",
            "system": "You are a helpful assistant.",
            "tools": [
                {
                    "type": "agent_toolset_20260401",
                    "configs": [
                        {"name": "web_search", "enabled": True, "permission_policy": {"type": "always_allow"}},
                        {"name": "bash", "enabled": False},
                    ],
                    "default_config": {"enabled": True, "permission_policy": {"type": "always_ask"}},
                },
                {
                    "type": "custom",
                    "name": "calculate",
                    "description": "Perform a calculation",
                    "input_schema": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    },
                },
            ],
        },
    )

    assert response.status_code == 200
    agent = response.json()
    assert len(agent["tools"]) == 2
    assert agent["tools"][0]["type"] == "agent_toolset_20260401"
    assert len(agent["tools"][0]["configs"]) == 2
    assert agent["tools"][1]["type"] == "custom"
    assert agent["tools"][1]["name"] == "calculate"


def test_create_agent_with_mcp_servers(agents_client, text_model_id):
    """Create an agent with MCP servers and MCP toolsets."""
    response = agents_client.post(
        "/agents",
        json={
            "model": text_model_id,
            "name": "MCP Agent",
            "description": "Agent with MCP integration",
            "mcp_servers": [{"name": "my-mcp-server", "type": "url", "url": "https://mcp.example.com"}],
            "tools": [
                {
                    "type": "mcp_toolset",
                    "mcp_server_name": "my-mcp-server",
                    "default_config": {"enabled": True, "permission_policy": {"type": "always_allow"}},
                }
            ],
        },
    )

    assert response.status_code == 200
    agent = response.json()
    assert len(agent["mcp_servers"]) == 1
    assert agent["mcp_servers"][0]["name"] == "my-mcp-server"
    assert len(agent["tools"]) == 1
    assert agent["tools"][0]["type"] == "mcp_toolset"
    assert agent["tools"][0]["mcp_server_name"] == "my-mcp-server"


def test_create_agent_with_metadata(agents_client, text_model_id):
    """Create an agent with skills and metadata."""
    response = agents_client.post(
        "/agents",
        json={
            "model": text_model_id,
            "name": "Metadata Agent",
            "skills": [
                {"type": "anthropic", "skill_id": "xlsx", "version": "1.0"},
            ],
            "metadata": {
                "owner": "test-user",
                "team": "engineering",
            },
        },
    )

    assert response.status_code == 200
    agent = response.json()
    assert len(agent["skills"]) == 1
    assert agent["skills"][0]["type"] == "anthropic"
    assert agent["skills"][0]["skill_id"] == "xlsx"
    assert agent["metadata"]["owner"] == "test-user"
    assert agent["metadata"]["team"] == "engineering"


def test_list_agents_empty(agents_client):
    """List agents returns empty list when no agents exist."""
    response = agents_client.get("/agents")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert isinstance(data["data"], list)


def test_get_agent(agents_client, text_model_id):
    """Create and retrieve an agent by ID."""
    # Create an agent
    create_response = agents_client.post(
        "/agents",
        json={
            "model": text_model_id,
            "name": "Get Test Agent",
            "system": "Test agent",
        },
    )
    agent_id = create_response.json()["id"]

    # Retrieve it
    get_response = agents_client.get(f"/agents/{agent_id}")
    assert get_response.status_code == 200
    agent = get_response.json()
    assert agent["id"] == agent_id
    assert agent["system"] == "Test agent"
    assert agent["name"] == "Get Test Agent"


def test_get_agent_not_found(agents_client):
    """Getting a non-existent agent returns 404."""
    response = agents_client.get("/agents/agent_nonexistent")
    assert response.status_code == 404
    error = response.json()
    assert error["type"] == "error"
    assert "not found" in error["error"]["message"].lower()


def test_update_agent(agents_client, text_model_id):
    """Update an agent creates a new version."""
    # Create an agent
    create_response = agents_client.post(
        "/agents",
        json={
            "model": text_model_id,
            "name": "Original Name",
            "system": "Original prompt",
            "skills": [{"type": "anthropic", "skill_id": "skill1"}],
        },
    )
    agent_id = create_response.json()["id"]

    # Update it
    update_response = agents_client.post(
        f"/agents/{agent_id}",
        json={
            "name": "Updated Name",
            "system": "Updated prompt",
            "skills": [
                {"type": "anthropic", "skill_id": "skill1"},
                {"type": "anthropic", "skill_id": "skill2"},
            ],
        },
    )

    assert update_response.status_code == 200
    updated_agent = update_response.json()
    assert updated_agent["id"] == agent_id
    assert updated_agent["version"] == 2  # Version incremented
    assert updated_agent["name"] == "Updated Name"
    assert updated_agent["system"] == "Updated prompt"
    assert len(updated_agent["skills"]) == 2
    assert updated_agent["model"] == text_model_id  # Model unchanged


def test_update_agent_not_found(agents_client):
    """Updating a non-existent agent returns 404."""
    response = agents_client.post(
        "/agents/agent_nonexistent",
        json={"system": "New prompt"},
    )
    assert response.status_code == 404


def test_archive_agent(agents_client, text_model_id):
    """Archive an agent sets archived_at timestamp."""
    # Create an agent
    create_response = agents_client.post(
        "/agents",
        json={"model": text_model_id, "name": "Archive Test"},
    )
    agent_id = create_response.json()["id"]

    # Archive it
    archive_response = agents_client.post(f"/agents/{agent_id}/archive")
    assert archive_response.status_code == 200
    archived_agent = archive_response.json()
    assert archived_agent["archived_at"] is not None
    assert archived_agent["id"] == agent_id


def test_archive_agent_idempotent(agents_client, text_model_id):
    """Archiving an already-archived agent succeeds."""
    # Create and archive an agent
    create_response = agents_client.post(
        "/agents",
        json={"model": text_model_id, "name": "Idempotent Test"},
    )
    agent_id = create_response.json()["id"]
    agents_client.post(f"/agents/{agent_id}/archive")

    # Archive again
    archive_response = agents_client.post(f"/agents/{agent_id}/archive")
    assert archive_response.status_code == 200
    assert archive_response.json()["archived_at"] is not None


def test_update_archived_agent_fails(agents_client, text_model_id):
    """Cannot update an archived agent."""
    # Create and archive an agent
    create_response = agents_client.post(
        "/agents",
        json={"model": text_model_id, "name": "Update Archived Test"},
    )
    agent_id = create_response.json()["id"]
    agents_client.post(f"/agents/{agent_id}/archive")

    # Try to update
    update_response = agents_client.post(
        f"/agents/{agent_id}",
        json={"system": "New prompt"},
    )
    assert update_response.status_code == 400
    error = update_response.json()
    assert "archived" in error["error"]["message"].lower()


def test_list_agents_pagination(agents_client, text_model_id):
    """List agents supports pagination with limit and after."""
    # Create multiple agents
    agent_ids = []
    for i in range(3):
        response = agents_client.post(
            "/agents",
            json={
                "model": text_model_id,
                "name": f"Agent {i}",
                "system": f"Agent {i}",
            },
        )
        agent_ids.append(response.json()["id"])

    # List with limit
    response = agents_client.get("/agents?limit=2")
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) <= 2

    # List with after cursor
    if len(data["data"]) > 0:
        first_id = data["data"][0]["id"]
        after_response = agents_client.get(f"/agents?after={first_id}")
        assert after_response.status_code == 200
        after_data = after_response.json()
        # Results after cursor should not include the cursor item
        assert all(a["id"] != first_id for a in after_data["data"])


def test_list_agents_filter_archived(agents_client, text_model_id):
    """List agents can filter by archived status."""
    # Create two agents, archive one
    response1 = agents_client.post("/agents", json={"model": text_model_id, "name": "Active Agent"})
    agent_id1 = response1.json()["id"]

    response2 = agents_client.post("/agents", json={"model": text_model_id, "name": "Archived Agent"})
    agent_id2 = response2.json()["id"]
    agents_client.post(f"/agents/{agent_id2}/archive")

    # List archived agents
    archived_response = agents_client.get("/agents?archived=true")
    assert archived_response.status_code == 200
    archived_data = archived_response.json()
    archived_ids = [a["id"] for a in archived_data["data"]]
    assert agent_id2 in archived_ids

    # List non-archived agents
    active_response = agents_client.get("/agents?archived=false")
    assert active_response.status_code == 200
    active_data = active_response.json()
    active_ids = [a["id"] for a in active_data["data"]]
    assert agent_id1 in active_ids
    assert agent_id2 not in active_ids


@pytest.mark.parametrize(
    "invalid_body,expected_error_substring",
    [
        ({}, "model"),  # Missing required field model
        ({"model": "test"}, "name"),  # Missing required field name
        ({"model": "test", "name": "x" * 257}, "name"),  # Name too long
        ({"model": "test", "name": "Test", "system": "x" * 100001}, "system"),  # System too long
    ],
)
def test_create_agent_validation_errors(agents_client, invalid_body, expected_error_substring):
    """Create agent with invalid body returns 400."""
    response = agents_client.post("/agents", json=invalid_body)
    assert response.status_code == 400
    error_text = response.text.lower()
    assert expected_error_substring.lower() in error_text

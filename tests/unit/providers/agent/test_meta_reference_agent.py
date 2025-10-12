# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from llama_stack.apis.agents import (
    Agent,
    AgentConfig,
    AgentCreateResponse,
)
from llama_stack.apis.common.responses import PaginatedResponse
from llama_stack.apis.conversations import Conversations
from llama_stack.apis.inference import Inference
from llama_stack.apis.safety import Safety
from llama_stack.apis.tools import ListToolDefsResponse, ToolDef, ToolGroups, ToolRuntime
from llama_stack.apis.vector_io import VectorIO
from llama_stack.providers.inline.agents.meta_reference.agent_instance import ChatAgent
from llama_stack.providers.inline.agents.meta_reference.agents import MetaReferenceAgentsImpl
from llama_stack.providers.inline.agents.meta_reference.config import MetaReferenceAgentsImplConfig
from llama_stack.providers.inline.agents.meta_reference.persistence import AgentInfo


@pytest.fixture
def mock_apis():
    return {
        "inference_api": AsyncMock(spec=Inference),
        "vector_io_api": AsyncMock(spec=VectorIO),
        "safety_api": AsyncMock(spec=Safety),
        "tool_runtime_api": AsyncMock(spec=ToolRuntime),
        "tool_groups_api": AsyncMock(spec=ToolGroups),
        "conversations_api": AsyncMock(spec=Conversations),
    }


@pytest.fixture
def config(tmp_path):
    return MetaReferenceAgentsImplConfig(
        persistence_store={
            "type": "sqlite",
            "db_path": str(tmp_path / "test.db"),
        },
        responses_store={
            "type": "sqlite",
            "db_path": str(tmp_path / "test.db"),
        },
    )


@pytest.fixture
async def agents_impl(config, mock_apis):
    impl = MetaReferenceAgentsImpl(
        config,
        mock_apis["inference_api"],
        mock_apis["vector_io_api"],
        mock_apis["safety_api"],
        mock_apis["tool_runtime_api"],
        mock_apis["tool_groups_api"],
        mock_apis["conversations_api"],
        [],
    )
    await impl.initialize()
    yield impl
    await impl.shutdown()


@pytest.fixture
def sample_agent_config():
    return AgentConfig(
        sampling_params={
            "strategy": {"type": "greedy"},
            "max_tokens": 0,
            "repetition_penalty": 1.0,
        },
        input_shields=["string"],
        output_shields=["string"],
        toolgroups=["mcp::my_mcp_server"],
        client_tools=[
            {
                "name": "client_tool",
                "description": "Client Tool",
                "parameters": [
                    {
                        "name": "string",
                        "parameter_type": "string",
                        "description": "string",
                        "required": True,
                        "default": None,
                    }
                ],
                "metadata": {
                    "property1": None,
                    "property2": None,
                },
            }
        ],
        tool_choice="auto",
        tool_prompt_format="json",
        tool_config={
            "tool_choice": "auto",
            "tool_prompt_format": "json",
            "system_message_behavior": "append",
        },
        max_infer_iters=10,
        model="string",
        instructions="string",
        enable_session_persistence=False,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "property1": None,
                "property2": None,
            },
        },
    )


async def test_create_agent(agents_impl, sample_agent_config):
    response = await agents_impl.create_agent(sample_agent_config)

    assert isinstance(response, AgentCreateResponse)
    assert response.agent_id is not None

    stored_agent = await agents_impl.persistence_store.get(f"agent:{response.agent_id}")
    assert stored_agent is not None
    agent_info = AgentInfo.model_validate_json(stored_agent)
    assert agent_info.model == sample_agent_config.model
    assert agent_info.created_at is not None
    assert isinstance(agent_info.created_at, datetime)


async def test_get_agent(agents_impl, sample_agent_config):
    create_response = await agents_impl.create_agent(sample_agent_config)
    agent_id = create_response.agent_id

    agent = await agents_impl.get_agent(agent_id)

    assert isinstance(agent, Agent)
    assert agent.agent_id == agent_id
    assert agent.agent_config.model == sample_agent_config.model
    assert agent.created_at is not None
    assert isinstance(agent.created_at, datetime)


async def test_list_agents(agents_impl, sample_agent_config):
    agent1_response = await agents_impl.create_agent(sample_agent_config)
    agent2_response = await agents_impl.create_agent(sample_agent_config)

    response = await agents_impl.list_agents()

    assert isinstance(response, PaginatedResponse)
    assert len(response.data) == 2
    agent_ids = {agent["agent_id"] for agent in response.data}
    assert agent1_response.agent_id in agent_ids
    assert agent2_response.agent_id in agent_ids


@pytest.mark.parametrize("enable_session_persistence", [True, False])
async def test_create_agent_session_persistence(agents_impl, sample_agent_config, enable_session_persistence):
    # Create an agent with specified persistence setting
    config = sample_agent_config.model_copy()
    config.enable_session_persistence = enable_session_persistence
    response = await agents_impl.create_agent(config)
    agent_id = response.agent_id

    # Create a session
    session_response = await agents_impl.create_agent_session(agent_id, "test_session")
    assert session_response.session_id is not None

    # Verify the session was stored
    session = await agents_impl.get_agents_session(agent_id, session_response.session_id)
    assert session.session_name == "test_session"
    assert session.session_id == session_response.session_id
    assert session.started_at is not None
    assert session.turns == []

    # Delete the session
    await agents_impl.delete_agents_session(agent_id, session_response.session_id)

    # Verify the session was deleted
    with pytest.raises(ValueError):
        await agents_impl.get_agents_session(agent_id, session_response.session_id)


@pytest.mark.parametrize("enable_session_persistence", [True, False])
async def test_list_agent_sessions_persistence(agents_impl, sample_agent_config, enable_session_persistence):
    # Create an agent with specified persistence setting
    config = sample_agent_config.model_copy()
    config.enable_session_persistence = enable_session_persistence
    response = await agents_impl.create_agent(config)
    agent_id = response.agent_id

    # Create multiple sessions
    session1 = await agents_impl.create_agent_session(agent_id, "session1")
    session2 = await agents_impl.create_agent_session(agent_id, "session2")

    # List sessions
    sessions = await agents_impl.list_agent_sessions(agent_id)
    assert len(sessions.data) == 2
    session_ids = {s["session_id"] for s in sessions.data}
    assert session1.session_id in session_ids
    assert session2.session_id in session_ids

    # Delete one session
    await agents_impl.delete_agents_session(agent_id, session1.session_id)

    # Verify the session was deleted
    with pytest.raises(ValueError):
        await agents_impl.get_agents_session(agent_id, session1.session_id)

    # List sessions again
    sessions = await agents_impl.list_agent_sessions(agent_id)
    assert len(sessions.data) == 1
    assert session2.session_id in {s["session_id"] for s in sessions.data}


async def test_delete_agent(agents_impl, sample_agent_config):
    # Create an agent
    response = await agents_impl.create_agent(sample_agent_config)
    agent_id = response.agent_id

    # Delete the agent
    await agents_impl.delete_agent(agent_id)

    # Verify the agent was deleted
    with pytest.raises(ValueError):
        await agents_impl.get_agent(agent_id)


async def test__initialize_tools(agents_impl, sample_agent_config):
    # Mock tool_groups_api.list_tools()
    agents_impl.tool_groups_api.list_tools.return_value = ListToolDefsResponse(
        data=[
            ToolDef(
                name="story_maker",
                toolgroup_id="mcp::my_mcp_server",
                description="Make a story",
                input_schema={
                    "type": "object",
                    "properties": {
                        "story_title": {"type": "string", "description": "Title of the story", "title": "Story Title"},
                        "input_words": {
                            "type": "array",
                            "description": "Input words",
                            "items": {"type": "string"},
                            "title": "Input Words",
                            "default": [],
                        },
                    },
                    "required": ["story_title"],
                },
            )
        ]
    )

    create_response = await agents_impl.create_agent(sample_agent_config)
    agent_id = create_response.agent_id

    # Get an instance of ChatAgent
    chat_agent = await agents_impl._get_agent_impl(agent_id)
    assert chat_agent is not None
    assert isinstance(chat_agent, ChatAgent)

    # Initialize tool definitions
    await chat_agent._initialize_tools()
    assert len(chat_agent.tool_defs) == 2

    # Verify the first tool, which is a client tool
    first_tool = chat_agent.tool_defs[0]
    assert first_tool.tool_name == "client_tool"
    assert first_tool.description == "Client Tool"

    # Verify the second tool, which is an MCP tool that has an array-type property
    second_tool = chat_agent.tool_defs[1]
    assert second_tool.tool_name == "story_maker"
    assert second_tool.description == "Make a story"

    # Verify the input schema
    input_schema = second_tool.input_schema
    assert input_schema is not None
    assert input_schema["type"] == "object"

    properties = input_schema["properties"]
    assert len(properties) == 2

    # Verify a string property
    story_title = properties["story_title"]
    assert story_title["type"] == "string"
    assert story_title["description"] == "Title of the story"
    assert story_title["title"] == "Story Title"

    # Verify an array property
    input_words = properties["input_words"]
    assert input_words["type"] == "array"
    assert input_words["description"] == "Input words"
    assert input_words["items"]["type"] == "string"
    assert input_words["title"] == "Input Words"
    assert input_words["default"] == []

    # Verify required fields
    assert input_schema["required"] == ["story_title"]

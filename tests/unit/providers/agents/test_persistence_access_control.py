# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import uuid
from datetime import datetime
from unittest.mock import patch

import pytest

from llama_stack.apis.agents import Turn
from llama_stack.apis.inference import CompletionMessage, StopReason
from llama_stack.core.datatypes import User
from llama_stack.providers.inline.agents.meta_reference.persistence import AgentPersistence, AgentSessionInfo


@pytest.fixture
async def test_setup(sqlite_kvstore):
    agent_persistence = AgentPersistence(agent_id="test_agent", kvstore=sqlite_kvstore, policy={})
    yield agent_persistence


@patch("llama_stack.providers.inline.agents.meta_reference.persistence.get_authenticated_user")
async def test_session_creation_with_access_attributes(mock_get_authenticated_user, test_setup):
    agent_persistence = test_setup

    # Set creator's attributes for the session
    creator_attributes = {"roles": ["researcher"], "teams": ["ai-team"]}
    mock_get_authenticated_user.return_value = User("test_user", creator_attributes)

    # Create a session
    session_id = await agent_persistence.create_session("Test Session")

    # Get the session and verify access attributes were set
    session_info = await agent_persistence.get_session_info(session_id)
    assert session_info is not None
    assert session_info.owner is not None
    assert session_info.owner.attributes is not None
    assert session_info.owner.attributes["roles"] == ["researcher"]
    assert session_info.owner.attributes["teams"] == ["ai-team"]


@patch("llama_stack.providers.inline.agents.meta_reference.persistence.get_authenticated_user")
async def test_session_access_control(mock_get_authenticated_user, test_setup):
    agent_persistence = test_setup

    # Create a session with specific access attributes
    session_id = str(uuid.uuid4())
    session_info = AgentSessionInfo(
        session_id=session_id,
        session_name="Restricted Session",
        started_at=datetime.now(),
        owner=User("someone", {"roles": ["admin"], "teams": ["security-team"]}),
        turns=[],
        identifier="Restricted Session",
    )

    await agent_persistence.kvstore.set(
        key=f"session:{agent_persistence.agent_id}:{session_id}",
        value=session_info.model_dump_json(),
    )

    # User with matching attributes can access
    mock_get_authenticated_user.return_value = User(
        "testuser", {"roles": ["admin", "user"], "teams": ["security-team", "other-team"]}
    )
    retrieved_session = await agent_persistence.get_session_info(session_id)
    assert retrieved_session is not None
    assert retrieved_session.session_id == session_id

    # User without matching attributes cannot access
    mock_get_authenticated_user.return_value = User("testuser", {"roles": ["user"], "teams": ["other-team"]})
    retrieved_session = await agent_persistence.get_session_info(session_id)
    assert retrieved_session is None


@patch("llama_stack.providers.inline.agents.meta_reference.persistence.get_authenticated_user")
async def test_turn_access_control(mock_get_authenticated_user, test_setup):
    agent_persistence = test_setup

    # Create a session with restricted access
    session_id = str(uuid.uuid4())
    session_info = AgentSessionInfo(
        session_id=session_id,
        session_name="Restricted Session",
        started_at=datetime.now(),
        owner=User("someone", {"roles": ["admin"]}),
        turns=[],
        identifier="Restricted Session",
    )

    await agent_persistence.kvstore.set(
        key=f"session:{agent_persistence.agent_id}:{session_id}",
        value=session_info.model_dump_json(),
    )

    # Create a turn for this session
    turn_id = str(uuid.uuid4())
    turn = Turn(
        session_id=session_id,
        turn_id=turn_id,
        steps=[],
        started_at=datetime.now(),
        input_messages=[],
        output_message=CompletionMessage(
            content="Hello",
            stop_reason=StopReason.end_of_turn,
        ),
    )

    # Admin can add turn
    mock_get_authenticated_user.return_value = User("testuser", {"roles": ["admin"]})
    await agent_persistence.add_turn_to_session(session_id, turn)

    # Admin can get turn
    retrieved_turn = await agent_persistence.get_session_turn(session_id, turn_id)
    assert retrieved_turn is not None
    assert retrieved_turn.turn_id == turn_id

    # Regular user cannot get turn
    mock_get_authenticated_user.return_value = User("testuser", {"roles": ["user"]})
    with pytest.raises(ValueError):
        await agent_persistence.get_session_turn(session_id, turn_id)

    # Regular user cannot get turns for session
    with pytest.raises(ValueError):
        await agent_persistence.get_session_turns(session_id)


@patch("llama_stack.providers.inline.agents.meta_reference.persistence.get_authenticated_user")
async def test_tool_call_and_infer_iters_access_control(mock_get_authenticated_user, test_setup):
    agent_persistence = test_setup

    # Create a session with restricted access
    session_id = str(uuid.uuid4())
    session_info = AgentSessionInfo(
        session_id=session_id,
        session_name="Restricted Session",
        started_at=datetime.now(),
        owner=User("someone", {"roles": ["admin"]}),
        turns=[],
        identifier="Restricted Session",
    )

    await agent_persistence.kvstore.set(
        key=f"session:{agent_persistence.agent_id}:{session_id}",
        value=session_info.model_dump_json(),
    )

    turn_id = str(uuid.uuid4())

    # Admin user can set inference iterations
    mock_get_authenticated_user.return_value = User("testuser", {"roles": ["admin"]})
    await agent_persistence.set_num_infer_iters_in_turn(session_id, turn_id, 5)

    # Admin user can get inference iterations
    infer_iters = await agent_persistence.get_num_infer_iters_in_turn(session_id, turn_id)
    assert infer_iters == 5

    # Regular user cannot get inference iterations
    mock_get_authenticated_user.return_value = User("testuser", {"roles": ["user"]})
    infer_iters = await agent_persistence.get_num_infer_iters_in_turn(session_id, turn_id)
    assert infer_iters is None

    # Regular user cannot set inference iterations (should raise ValueError)
    with pytest.raises(ValueError):
        await agent_persistence.set_num_infer_iters_in_turn(session_id, turn_id, 10)

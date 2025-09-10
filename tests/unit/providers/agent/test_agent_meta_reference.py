# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from llama_stack.apis.agents import Session
from llama_stack.core.datatypes import User
from llama_stack.providers.inline.agents.meta_reference.persistence import (
    AgentPersistence,
    AgentSessionInfo,
)
from llama_stack.providers.utils.kvstore import KVStore


@pytest.fixture
def mock_kvstore():
    return AsyncMock(spec=KVStore)


@pytest.fixture
def mock_policy():
    return []


@pytest.fixture
def agent_persistence(mock_kvstore, mock_policy):
    return AgentPersistence(agent_id="test-agent-123", kvstore=mock_kvstore, policy=mock_policy)


@pytest.fixture
def sample_session():
    return AgentSessionInfo(
        session_id="session-123",
        session_name="Test Session",
        started_at=datetime.now(UTC),
        owner=User(principal="user-123", attributes=None),
        turns=[],
        identifier="test-session",
        type="session",
    )


@pytest.fixture
def sample_session_json(sample_session):
    return sample_session.model_dump_json()


class TestAgentPersistenceListSessions:
    def setup_mock_kvstore(self, mock_kvstore, session_keys=None, turn_keys=None, invalid_keys=None, custom_data=None):
        """Helper to setup mock kvstore with sessions, turns, and custom/invalid data

        Args:
            mock_kvstore: The mock KVStore object
            session_keys: List of session keys or dict mapping keys to custom session data
            turn_keys: List of turn keys or dict mapping keys to custom turn data
            invalid_keys: Dict mapping keys to invalid/corrupt data
            custom_data: Additional custom data to add to the mock responses
        """
        all_keys = []
        mock_data = {}

        # session keys
        if session_keys:
            if isinstance(session_keys, dict):
                all_keys.extend(session_keys.keys())
                mock_data.update({k: json.dumps(v) if isinstance(v, dict) else v for k, v in session_keys.items()})
            else:
                all_keys.extend(session_keys)
                for key in session_keys:
                    session_id = key.split(":")[-1]
                    mock_data[key] = json.dumps(
                        {
                            "session_id": session_id,
                            "session_name": f"Session {session_id}",
                            "started_at": datetime.now(UTC).isoformat(),
                            "turns": [],
                        }
                    )

        # turn keys
        if turn_keys:
            if isinstance(turn_keys, dict):
                all_keys.extend(turn_keys.keys())
                mock_data.update({k: json.dumps(v) if isinstance(v, dict) else v for k, v in turn_keys.items()})
            else:
                all_keys.extend(turn_keys)
                for key in turn_keys:
                    parts = key.split(":")
                    session_id = parts[-2]
                    turn_id = parts[-1]
                    mock_data[key] = json.dumps(
                        {
                            "turn_id": turn_id,
                            "session_id": session_id,
                            "input_messages": [],
                            "started_at": datetime.now(UTC).isoformat(),
                        }
                    )

        if invalid_keys:
            all_keys.extend(invalid_keys.keys())
            mock_data.update(invalid_keys)

        if custom_data:
            mock_data.update(custom_data)

        values_list = list(mock_data.values())
        mock_kvstore.values_in_range.return_value = values_list

        async def mock_get(key):
            return mock_data.get(key)

        mock_kvstore.get.side_effect = mock_get

        return mock_data

    @pytest.mark.parametrize(
        "scenario",
        [
            {
                # from this issue: https://github.com/meta-llama/llama-stack/issues/3048
                "name": "reported_bug",
                "session_keys": ["session:test-agent-123:1f08fd1c-5a9d-459d-a00b-36d4dfa49b7d"],
                "turn_keys": [
                    "session:test-agent-123:1f08fd1c-5a9d-459d-a00b-36d4dfa49b7d:eb7e818f-41fb-49a0-bdd6-464974a2d2ad"
                ],
                "expected_sessions": ["1f08fd1c-5a9d-459d-a00b-36d4dfa49b7d"],
            },
            {
                "name": "basic_filtering",
                "session_keys": ["session:test-agent-123:session-1", "session:test-agent-123:session-2"],
                "turn_keys": ["session:test-agent-123:session-1:turn-1", "session:test-agent-123:session-1:turn-2"],
                "expected_sessions": ["session-1", "session-2"],
            },
            {
                "name": "multiple_turns_per_session",
                "session_keys": ["session:test-agent-123:session-456"],
                "turn_keys": [
                    "session:test-agent-123:session-456:turn-789",
                    "session:test-agent-123:session-456:turn-790",
                ],
                "expected_sessions": ["session-456"],
            },
            {
                "name": "multiple_sessions_with_turns",
                "session_keys": ["session:test-agent-123:session-1", "session:test-agent-123:session-2"],
                "turn_keys": [
                    "session:test-agent-123:session-1:turn-1",
                    "session:test-agent-123:session-1:turn-2",
                    "session:test-agent-123:session-2:turn-3",
                ],
                "expected_sessions": ["session-1", "session-2"],
            },
        ],
    )
    async def test_list_sessions_key_filtering(self, agent_persistence, mock_kvstore, scenario):
        self.setup_mock_kvstore(mock_kvstore, session_keys=scenario["session_keys"], turn_keys=scenario["turn_keys"])

        with patch("llama_stack.providers.inline.agents.meta_reference.persistence.log") as mock_log:
            result = await agent_persistence.list_sessions()

        assert len(result) == len(scenario["expected_sessions"])
        session_ids = {s.session_id for s in result}
        for expected_id in scenario["expected_sessions"]:
            assert expected_id in session_ids

        # no errors should be logged
        mock_log.error.assert_not_called()

    @pytest.mark.parametrize(
        "error_scenario",
        [
            {
                "name": "invalid_json",
                "valid_keys": ["session:test-agent-123:valid-session"],
                "invalid_data": {"session:test-agent-123:invalid-json": "corrupted-json-data{"},
                "expected_valid_sessions": ["valid-session"],
                "expected_error_count": 1,
            },
            {
                "name": "missing_fields",
                "valid_keys": ["session:test-agent-123:valid-session"],
                "invalid_data": {
                    "session:test-agent-123:invalid-schema": json.dumps(
                        {
                            "session_id": "invalid-schema",
                            "session_name": "Missing Fields",
                            # missing `started_at` and `turns`
                        }
                    )
                },
                "expected_valid_sessions": ["valid-session"],
                "expected_error_count": 1,
            },
            {
                "name": "multiple_invalid",
                "valid_keys": ["session:test-agent-123:valid-session-1", "session:test-agent-123:valid-session-2"],
                "invalid_data": {
                    "session:test-agent-123:corrupted-json": "not-valid-json{",
                    "session:test-agent-123:incomplete-data": json.dumps({"incomplete": "data"}),
                },
                "expected_valid_sessions": ["valid-session-1", "valid-session-2"],
                "expected_error_count": 2,
            },
        ],
    )
    async def test_list_sessions_error_handling(self, agent_persistence, mock_kvstore, error_scenario):
        session_keys = {}
        for key in error_scenario["valid_keys"]:
            session_id = key.split(":")[-1]
            session_keys[key] = {
                "session_id": session_id,
                "session_name": f"Valid {session_id}",
                "started_at": datetime.now(UTC).isoformat(),
                "turns": [],
            }

        self.setup_mock_kvstore(mock_kvstore, session_keys=session_keys, invalid_keys=error_scenario["invalid_data"])

        with patch("llama_stack.providers.inline.agents.meta_reference.persistence.log") as mock_log:
            result = await agent_persistence.list_sessions()

        # only valid sessions should be returned
        assert len(result) == len(error_scenario["expected_valid_sessions"])
        session_ids = {s.session_id for s in result}
        for expected_id in error_scenario["expected_valid_sessions"]:
            assert expected_id in session_ids

        # error should be logged
        assert mock_log.error.call_count > 0
        assert mock_log.error.call_count == error_scenario["expected_error_count"]

    async def test_list_sessions_empty(self, agent_persistence, mock_kvstore):
        mock_kvstore.values_in_range.return_value = []

        result = await agent_persistence.list_sessions()

        assert result == []
        mock_kvstore.values_in_range.assert_called_once_with(
            start_key="session:test-agent-123:", end_key="session:test-agent-123:\xff\xff\xff\xff"
        )

    async def test_list_sessions_properties(self, agent_persistence, mock_kvstore):
        session_data = {
            "session_id": "session-123",
            "session_name": "Test Session",
            "started_at": datetime.now(UTC).isoformat(),
            "owner": {"principal": "user-123", "attributes": None},
            "turns": [],
        }

        self.setup_mock_kvstore(mock_kvstore, session_keys={"session:test-agent-123:session-123": session_data})

        result = await agent_persistence.list_sessions()

        assert len(result) == 1
        assert isinstance(result[0], Session)
        assert result[0].session_id == "session-123"
        assert result[0].session_name == "Test Session"
        assert result[0].turns == []
        assert hasattr(result[0], "started_at")

    async def test_list_sessions_kvstore_exception(self, agent_persistence, mock_kvstore):
        mock_kvstore.values_in_range.side_effect = Exception("KVStore error")

        with pytest.raises(Exception, match="KVStore error"):
            await agent_persistence.list_sessions()

    async def test_bug_data_loss_with_real_data(self, agent_persistence, mock_kvstore):
        # tests the handling of the issue reported in: https://github.com/meta-llama/llama-stack/issues/3048
        session_data = {
            "session_id": "1f08fd1c-5a9d-459d-a00b-36d4dfa49b7d",
            "session_name": "Test Session",
            "started_at": datetime.now(UTC).isoformat(),
            "turns": [],
        }

        turn_data = {
            "turn_id": "eb7e818f-41fb-49a0-bdd6-464974a2d2ad",
            "session_id": "1f08fd1c-5a9d-459d-a00b-36d4dfa49b7d",
            "input_messages": [
                {"role": "user", "content": "if i had a cluster i would want to call it persistence01", "context": None}
            ],
            "steps": [
                {
                    "turn_id": "eb7e818f-41fb-49a0-bdd6-464974a2d2ad",
                    "step_id": "c0f797dd-3d34-4bc5-a8f4-db6af9455132",
                    "started_at": "2025-08-05T14:31:50.000484Z",
                    "completed_at": "2025-08-05T14:31:51.303691Z",
                    "step_type": "inference",
                    "model_response": {
                        "role": "assistant",
                        "content": "OK, I can create a cluster named 'persistence01' for you.",
                        "stop_reason": "end_of_turn",
                        "tool_calls": [],
                    },
                }
            ],
            "output_message": {
                "role": "assistant",
                "content": "OK, I can create a cluster named 'persistence01' for you.",
                "stop_reason": "end_of_turn",
                "tool_calls": [],
            },
            "output_attachments": [],
            "started_at": "2025-08-05T14:31:49.999950Z",
            "completed_at": "2025-08-05T14:31:51.305384Z",
        }

        mock_data = {
            "session:test-agent-123:1f08fd1c-5a9d-459d-a00b-36d4dfa49b7d": json.dumps(session_data),
            "session:test-agent-123:1f08fd1c-5a9d-459d-a00b-36d4dfa49b7d:eb7e818f-41fb-49a0-bdd6-464974a2d2ad": json.dumps(
                turn_data
            ),
        }

        mock_kvstore.values_in_range.return_value = list(mock_data.values())

        async def mock_get(key):
            return mock_data.get(key)

        mock_kvstore.get.side_effect = mock_get

        with patch("llama_stack.providers.inline.agents.meta_reference.persistence.log") as mock_log:
            result = await agent_persistence.list_sessions()

        assert len(result) == 1
        assert result[0].session_id == "1f08fd1c-5a9d-459d-a00b-36d4dfa49b7d"

        # confirm no errors logged
        mock_log.error.assert_not_called()

    async def test_list_sessions_key_range_construction(self, agent_persistence, mock_kvstore):
        mock_kvstore.values_in_range.return_value = []

        await agent_persistence.list_sessions()

        mock_kvstore.values_in_range.assert_called_once_with(
            start_key="session:test-agent-123:", end_key="session:test-agent-123:\xff\xff\xff\xff"
        )

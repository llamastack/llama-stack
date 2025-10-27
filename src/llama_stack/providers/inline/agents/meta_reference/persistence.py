# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import uuid
from datetime import UTC, datetime

from llama_stack.apis.agents import AgentConfig, Session, ToolExecutionStep, Turn
from llama_stack.apis.common.errors import SessionNotFoundError
from llama_stack.core.access_control.access_control import AccessDeniedError, is_action_allowed
from llama_stack.core.access_control.datatypes import AccessRule
from llama_stack.core.datatypes import User
from llama_stack.core.request_headers import get_authenticated_user
from llama_stack.log import get_logger
from llama_stack.providers.utils.kvstore import KVStore

log = get_logger(name=__name__, category="agents::meta_reference")


class AgentSessionInfo(Session):
    # TODO: is this used anywhere?
    vector_db_id: str | None = None
    started_at: datetime
    owner: User | None = None
    identifier: str | None = None
    type: str = "session"


class AgentInfo(AgentConfig):
    created_at: datetime


class AgentPersistence:
    def __init__(self, agent_id: str, kvstore: KVStore, policy: list[AccessRule]):
        self.agent_id = agent_id
        self.kvstore = kvstore
        self.policy = policy

    async def create_session(self, name: str) -> str:
        session_id = str(uuid.uuid4())

        # Get current user's auth attributes for new sessions
        user = get_authenticated_user()

        session_info = AgentSessionInfo(
            session_id=session_id,
            session_name=name,
            started_at=datetime.now(UTC),
            owner=user,
            turns=[],
            identifier=name,  # should this be qualified in any way?
        )
        if not is_action_allowed(self.policy, "create", session_info, user):
            raise AccessDeniedError("create", session_info, user)

        await self.kvstore.set(
            key=f"session:{self.agent_id}:{session_id}",
            value=session_info.model_dump_json(),
        )
        return session_id

    async def get_session_info(self, session_id: str) -> AgentSessionInfo:
        value = await self.kvstore.get(
            key=f"session:{self.agent_id}:{session_id}",
        )
        if not value:
            raise SessionNotFoundError(session_id)

        session_info = AgentSessionInfo(**json.loads(value))

        # Check access to session
        if not self._check_session_access(session_info):
            return None

        return session_info

    def _check_session_access(self, session_info: AgentSessionInfo) -> bool:
        """Check if current user has access to the session."""
        # Handle backward compatibility for old sessions without access control
        if not hasattr(session_info, "access_attributes") and not hasattr(session_info, "owner"):
            return True

        return is_action_allowed(self.policy, "read", session_info, get_authenticated_user())

    async def get_session_if_accessible(self, session_id: str) -> AgentSessionInfo | None:
        """Get session info if the user has access to it. For internal use by sub-session methods."""
        session_info = await self.get_session_info(session_id)
        if not session_info:
            return None

        return session_info

    async def add_vector_db_to_session(self, session_id: str, vector_db_id: str):
        session_info = await self.get_session_if_accessible(session_id)
        if session_info is None:
            raise SessionNotFoundError(session_id)

        session_info.vector_db_id = vector_db_id
        await self.kvstore.set(
            key=f"session:{self.agent_id}:{session_id}",
            value=session_info.model_dump_json(),
        )

    async def add_turn_to_session(self, session_id: str, turn: Turn):
        if not await self.get_session_if_accessible(session_id):
            raise SessionNotFoundError(session_id)

        await self.kvstore.set(
            key=f"session:{self.agent_id}:{session_id}:{turn.turn_id}",
            value=turn.model_dump_json(),
        )

    async def get_session_turns(self, session_id: str) -> list[Turn]:
        if not await self.get_session_if_accessible(session_id):
            raise SessionNotFoundError(session_id)

        values = await self.kvstore.values_in_range(
            start_key=f"session:{self.agent_id}:{session_id}:",
            end_key=f"session:{self.agent_id}:{session_id}:\xff\xff\xff\xff",
        )
        turns = []
        for value in values:
            try:
                turn = Turn(**json.loads(value))
                turns.append(turn)
            except Exception as e:
                log.error(f"Error parsing turn: {e}")
                continue

        # The kvstore does not guarantee order, so we sort by started_at
        # to ensure consistent ordering of turns.
        turns.sort(key=lambda t: t.started_at)

        return turns

    async def get_session_turn(self, session_id: str, turn_id: str) -> Turn | None:
        if not await self.get_session_if_accessible(session_id):
            raise SessionNotFoundError(session_id)

        value = await self.kvstore.get(
            key=f"session:{self.agent_id}:{session_id}:{turn_id}",
        )
        if not value:
            return None
        return Turn(**json.loads(value))

    async def set_in_progress_tool_call_step(self, session_id: str, turn_id: str, step: ToolExecutionStep):
        if not await self.get_session_if_accessible(session_id):
            raise SessionNotFoundError(session_id)

        await self.kvstore.set(
            key=f"in_progress_tool_call_step:{self.agent_id}:{session_id}:{turn_id}",
            value=step.model_dump_json(),
        )

    async def get_in_progress_tool_call_step(self, session_id: str, turn_id: str) -> ToolExecutionStep | None:
        if not await self.get_session_if_accessible(session_id):
            return None

        value = await self.kvstore.get(
            key=f"in_progress_tool_call_step:{self.agent_id}:{session_id}:{turn_id}",
        )
        return ToolExecutionStep(**json.loads(value)) if value else None

    async def set_num_infer_iters_in_turn(self, session_id: str, turn_id: str, num_infer_iters: int):
        if not await self.get_session_if_accessible(session_id):
            raise SessionNotFoundError(session_id)

        await self.kvstore.set(
            key=f"num_infer_iters_in_turn:{self.agent_id}:{session_id}:{turn_id}",
            value=str(num_infer_iters),
        )

    async def get_num_infer_iters_in_turn(self, session_id: str, turn_id: str) -> int | None:
        if not await self.get_session_if_accessible(session_id):
            return None

        value = await self.kvstore.get(
            key=f"num_infer_iters_in_turn:{self.agent_id}:{session_id}:{turn_id}",
        )
        return int(value) if value else None

    async def list_sessions(self) -> list[Session]:
        values = await self.kvstore.values_in_range(
            start_key=f"session:{self.agent_id}:",
            end_key=f"session:{self.agent_id}:\xff\xff\xff\xff",
        )
        sessions = []
        for value in values:
            try:
                data = json.loads(value)
                if "turn_id" in data:
                    continue

                session_info = Session(**data)
                sessions.append(session_info)
            except Exception as e:
                log.error(f"Error parsing session info: {e}")
                continue
        return sessions

    async def delete_session_turns(self, session_id: str) -> None:
        """Delete all turns and their associated data for a session.

        Args:
            session_id: The ID of the session whose turns should be deleted.
        """
        turns = await self.get_session_turns(session_id)
        for turn in turns:
            await self.kvstore.delete(key=f"session:{self.agent_id}:{session_id}:{turn.turn_id}")

    async def delete_session(self, session_id: str) -> None:
        """Delete a session and all its associated turns.

        Args:
            session_id: The ID of the session to delete.

        Raises:
            ValueError: If the session does not exist.
        """
        session_info = await self.get_session_info(session_id)
        if session_info is None:
            raise SessionNotFoundError(session_id)

        await self.kvstore.delete(key=f"session:{self.agent_id}:{session_id}")

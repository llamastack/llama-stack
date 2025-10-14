# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

from llama_stack.apis.agents import (
    Agent,
    AgentConfig,
    AgentCreateResponse,
    Agents,
    AgentSessionCreateResponse,
    AgentStepResponse,
    AgentToolGroup,
    AgentTurnCreateRequest,
    AgentTurnResumeRequest,
    Document,
    ListOpenAIResponseInputItem,
    ListOpenAIResponseObject,
    OpenAIResponseInput,
    OpenAIResponseInputTool,
    OpenAIResponseObject,
    Order,
    Session,
    Turn,
)
from llama_stack.apis.agents.agents import ResponseGuardrail
from llama_stack.apis.agents.openai_responses import OpenAIResponseText
from llama_stack.apis.common.responses import PaginatedResponse
from llama_stack.apis.conversations import Conversations
from llama_stack.apis.inference import (
    Inference,
    ToolConfig,
    ToolResponse,
    ToolResponseMessage,
    UserMessage,
)
from llama_stack.apis.safety import Safety
from llama_stack.apis.tools import ToolGroups, ToolRuntime
from llama_stack.apis.vector_io import VectorIO
from llama_stack.core.datatypes import AccessRule
from llama_stack.log import get_logger
from llama_stack.providers.utils.kvstore import InmemoryKVStoreImpl, kvstore_impl
from llama_stack.providers.utils.pagination import paginate_records
from llama_stack.providers.utils.responses.responses_store import ResponsesStore

from .agent_instance import ChatAgent
from .config import MetaReferenceAgentsImplConfig
from .persistence import AgentInfo
from .responses.openai_responses import OpenAIResponsesImpl

logger = get_logger(name=__name__, category="agents::meta_reference")


class MetaReferenceAgentsImpl(Agents):
    def __init__(
        self,
        config: MetaReferenceAgentsImplConfig,
        inference_api: Inference,
        vector_io_api: VectorIO,
        safety_api: Safety,
        tool_runtime_api: ToolRuntime,
        tool_groups_api: ToolGroups,
        conversations_api: Conversations,
        policy: list[AccessRule],
        telemetry_enabled: bool = False,
    ):
        self.config = config
        self.inference_api = inference_api
        self.vector_io_api = vector_io_api
        self.safety_api = safety_api
        self.tool_runtime_api = tool_runtime_api
        self.tool_groups_api = tool_groups_api
        self.conversations_api = conversations_api
        self.telemetry_enabled = telemetry_enabled

        self.in_memory_store = InmemoryKVStoreImpl()
        self.openai_responses_impl: OpenAIResponsesImpl | None = None
        self.policy = policy

    async def initialize(self) -> None:
        self.persistence_store = await kvstore_impl(self.config.persistence_store)
        self.responses_store = ResponsesStore(self.config.responses_store, self.policy)
        await self.responses_store.initialize()
        self.openai_responses_impl = OpenAIResponsesImpl(
            inference_api=self.inference_api,
            tool_groups_api=self.tool_groups_api,
            tool_runtime_api=self.tool_runtime_api,
            responses_store=self.responses_store,
            vector_io_api=self.vector_io_api,
            safety_api=self.safety_api,
            conversations_api=self.conversations_api,
        )

    async def create_agent(
        self,
        agent_config: AgentConfig,
    ) -> AgentCreateResponse:
        agent_id = str(uuid.uuid4())
        created_at = datetime.now(UTC)

        agent_info = AgentInfo(
            **agent_config.model_dump(),
            created_at=created_at,
        )

        # Store the agent info
        await self.persistence_store.set(
            key=f"agent:{agent_id}",
            value=agent_info.model_dump_json(),
        )

        return AgentCreateResponse(
            agent_id=agent_id,
        )

    async def _get_agent_impl(self, agent_id: str) -> ChatAgent:
        agent_info_json = await self.persistence_store.get(
            key=f"agent:{agent_id}",
        )
        if not agent_info_json:
            raise ValueError(f"Could not find agent info for {agent_id}")

        try:
            agent_info = AgentInfo.model_validate_json(agent_info_json)
        except Exception as e:
            raise ValueError(f"Could not validate agent info for {agent_id}") from e

        return ChatAgent(
            agent_id=agent_id,
            agent_config=agent_info,
            inference_api=self.inference_api,
            safety_api=self.safety_api,
            vector_io_api=self.vector_io_api,
            tool_runtime_api=self.tool_runtime_api,
            tool_groups_api=self.tool_groups_api,
            persistence_store=(
                self.persistence_store if agent_info.enable_session_persistence else self.in_memory_store
            ),
            created_at=agent_info.created_at,
            policy=self.policy,
            telemetry_enabled=self.telemetry_enabled,
        )

    async def create_agent_session(
        self,
        agent_id: str,
        session_name: str,
    ) -> AgentSessionCreateResponse:
        agent = await self._get_agent_impl(agent_id)

        session_id = await agent.create_session(session_name)
        return AgentSessionCreateResponse(
            session_id=session_id,
        )

    async def create_agent_turn(
        self,
        agent_id: str,
        session_id: str,
        messages: list[UserMessage | ToolResponseMessage],
        toolgroups: list[AgentToolGroup] | None = None,
        documents: list[Document] | None = None,
        stream: bool | None = False,
        tool_config: ToolConfig | None = None,
    ) -> AsyncGenerator:
        request = AgentTurnCreateRequest(
            agent_id=agent_id,
            session_id=session_id,
            messages=messages,
            stream=True,
            toolgroups=toolgroups,
            documents=documents,
            tool_config=tool_config,
        )
        if stream:
            return self._create_agent_turn_streaming(request)
        else:
            raise NotImplementedError("Non-streaming agent turns not yet implemented")

    async def _create_agent_turn_streaming(
        self,
        request: AgentTurnCreateRequest,
    ) -> AsyncGenerator:
        agent = await self._get_agent_impl(request.agent_id)
        async for event in agent.create_and_execute_turn(request):
            yield event

    async def resume_agent_turn(
        self,
        agent_id: str,
        session_id: str,
        turn_id: str,
        tool_responses: list[ToolResponse],
        stream: bool | None = False,
    ) -> AsyncGenerator:
        request = AgentTurnResumeRequest(
            agent_id=agent_id,
            session_id=session_id,
            turn_id=turn_id,
            tool_responses=tool_responses,
            stream=stream,
        )
        if stream:
            return self._continue_agent_turn_streaming(request)
        else:
            raise NotImplementedError("Non-streaming agent turns not yet implemented")

    async def _continue_agent_turn_streaming(
        self,
        request: AgentTurnResumeRequest,
    ) -> AsyncGenerator:
        agent = await self._get_agent_impl(request.agent_id)
        async for event in agent.resume_turn(request):
            yield event

    async def get_agents_turn(self, agent_id: str, session_id: str, turn_id: str) -> Turn:
        agent = await self._get_agent_impl(agent_id)
        turn = await agent.storage.get_session_turn(session_id, turn_id)
        return turn

    async def get_agents_step(self, agent_id: str, session_id: str, turn_id: str, step_id: str) -> AgentStepResponse:
        turn = await self.get_agents_turn(agent_id, session_id, turn_id)
        for step in turn.steps:
            if step.step_id == step_id:
                return AgentStepResponse(step=step)
        raise ValueError(f"Provided step_id {step_id} could not be found")

    async def get_agents_session(
        self,
        agent_id: str,
        session_id: str,
        turn_ids: list[str] | None = None,
    ) -> Session:
        agent = await self._get_agent_impl(agent_id)

        session_info = await agent.storage.get_session_info(session_id)
        turns = await agent.storage.get_session_turns(session_id)
        if turn_ids:
            turns = [turn for turn in turns if turn.turn_id in turn_ids]
        return Session(
            session_name=session_info.session_name,
            session_id=session_id,
            turns=turns,
            started_at=session_info.started_at,
        )

    async def delete_agents_session(self, agent_id: str, session_id: str) -> None:
        agent = await self._get_agent_impl(agent_id)

        # Delete turns first, then the session
        await agent.storage.delete_session_turns(session_id)
        await agent.storage.delete_session(session_id)

    async def delete_agent(self, agent_id: str) -> None:
        # First get all sessions for this agent
        agent = await self._get_agent_impl(agent_id)
        sessions = await agent.storage.list_sessions()

        # Delete all sessions
        for session in sessions:
            await self.delete_agents_session(agent_id, session.session_id)

        # Finally delete the agent itself
        await self.persistence_store.delete(f"agent:{agent_id}")

    async def list_agents(self, start_index: int | None = None, limit: int | None = None) -> PaginatedResponse:
        agent_keys = await self.persistence_store.keys_in_range("agent:", "agent:\xff")
        agent_list: list[Agent] = []
        for agent_key in agent_keys:
            agent_id = agent_key.split(":")[1]

            # Get the agent info using the key
            agent_info_json = await self.persistence_store.get(agent_key)
            if not agent_info_json:
                logger.error(f"Could not find agent info for key {agent_key}")
                continue

            try:
                agent_info = AgentInfo.model_validate_json(agent_info_json)
                agent_list.append(
                    Agent(
                        agent_id=agent_id,
                        agent_config=agent_info,
                        created_at=agent_info.created_at,
                    )
                )
            except Exception as e:
                logger.error(f"Error parsing agent info for {agent_id}: {e}")
                continue

        # Convert Agent objects to dictionaries
        agent_dicts = [agent.model_dump() for agent in agent_list]
        return paginate_records(agent_dicts, start_index, limit)

    async def get_agent(self, agent_id: str) -> Agent:
        chat_agent = await self._get_agent_impl(agent_id)
        agent = Agent(
            agent_id=agent_id,
            agent_config=chat_agent.agent_config,
            created_at=chat_agent.created_at,
        )
        return agent

    async def list_agent_sessions(
        self, agent_id: str, start_index: int | None = None, limit: int | None = None
    ) -> PaginatedResponse:
        agent = await self._get_agent_impl(agent_id)
        sessions = await agent.storage.list_sessions()
        # Convert Session objects to dictionaries
        session_dicts = [session.model_dump() for session in sessions]
        return paginate_records(session_dicts, start_index, limit)

    async def shutdown(self) -> None:
        pass

    # OpenAI responses
    async def get_openai_response(
        self,
        response_id: str,
    ) -> OpenAIResponseObject:
        return await self.openai_responses_impl.get_openai_response(response_id)

    async def create_openai_response(
        self,
        input: str | list[OpenAIResponseInput],
        model: str,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        conversation: str | None = None,
        store: bool | None = True,
        stream: bool | None = False,
        temperature: float | None = None,
        text: OpenAIResponseText | None = None,
        tools: list[OpenAIResponseInputTool] | None = None,
        include: list[str] | None = None,
        max_infer_iters: int | None = 10,
        guardrails: list[ResponseGuardrail] | None = None,
    ) -> OpenAIResponseObject:
        return await self.openai_responses_impl.create_openai_response(
            input,
            model,
            instructions,
            previous_response_id,
            conversation,
            store,
            stream,
            temperature,
            text,
            tools,
            include,
            max_infer_iters,
            guardrails,
        )

    async def list_openai_responses(
        self,
        after: str | None = None,
        limit: int | None = 50,
        model: str | None = None,
        order: Order | None = Order.desc,
    ) -> ListOpenAIResponseObject:
        return await self.openai_responses_impl.list_openai_responses(after, limit, model, order)

    async def list_openai_response_input_items(
        self,
        response_id: str,
        after: str | None = None,
        before: str | None = None,
        include: list[str] | None = None,
        limit: int | None = 20,
        order: Order | None = Order.desc,
    ) -> ListOpenAIResponseInputItem:
        return await self.openai_responses_impl.list_openai_response_input_items(
            response_id, after, before, include, limit, order
        )

    async def delete_openai_response(self, response_id: str) -> None:
        return await self.openai_responses_impl.delete_openai_response(response_id)

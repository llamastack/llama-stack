# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated

from fastapi import Body, Depends, Query, Request
from fastapi import Path as FastAPIPath

from llama_stack.apis.common.responses import Order
from llama_stack.apis.datatypes import Api
from llama_stack.apis.version import LLAMA_STACK_API_V1, LLAMA_STACK_API_V1ALPHA
from llama_stack.core.server.router_utils import standard_responses
from llama_stack.core.server.routers import APIRouter, register_router

from .agents_service import AgentsService
from .models import (
    Agent,
    AgentConfig,
    AgentCreateResponse,
    AgentSessionCreateResponse,
    AgentStepResponse,
    AgentTurnCreateRequest,
    AgentTurnResumeRequest,
    CreateAgentSessionRequest,
    CreateOpenAIResponseRequest,
    Session,
    Turn,
)
from .openai_responses import (
    ListOpenAIResponseInputItem,
    ListOpenAIResponseObject,
    OpenAIDeleteResponseObject,
    OpenAIResponseObject,
)


def get_agents_service(request: Request) -> AgentsService:
    """Dependency to get the agents service implementation from app state."""
    impls = getattr(request.app.state, "impls", {})
    if Api.agents not in impls:
        raise ValueError("Agents API implementation not found")
    return impls[Api.agents]


router = APIRouter(
    prefix=f"/{LLAMA_STACK_API_V1}",
    tags=["Agents"],
    responses=standard_responses,
)

router_v1alpha = APIRouter(
    prefix=f"/{LLAMA_STACK_API_V1ALPHA}",
    tags=["Agents"],
    responses=standard_responses,
)


@router.post(
    "/agents",
    response_model=AgentCreateResponse,
    summary="Create an agent.",
    description="Create an agent with the given configuration.",
    deprecated=True,
)
@router_v1alpha.post(
    "/agents",
    response_model=AgentCreateResponse,
    summary="Create an agent.",
    description="Create an agent with the given configuration.",
)
async def create_agent(
    agent_config: AgentConfig = Body(...),
    svc: AgentsService = Depends(get_agents_service),
) -> AgentCreateResponse:
    """Create an agent with the given configuration."""
    return await svc.create_agent(agent_config=agent_config)


@router.post(
    "/agents/{agent_id}/session/{session_id}/turn",
    summary="Create a new turn for an agent.",
    description="Create a new turn for an agent.",
    deprecated=True,
)
@router_v1alpha.post(
    "/agents/{{agent_id}}/session/{{session_id}}/turn",
    summary="Create a new turn for an agent.",
    description="Create a new turn for an agent.",
)
async def create_agent_turn(
    agent_id: Annotated[str, FastAPIPath(..., description="The ID of the agent to create the turn for.")],
    session_id: Annotated[str, FastAPIPath(..., description="The ID of the session to create the turn for.")],
    body: AgentTurnCreateRequest = Body(...),
    svc: AgentsService = Depends(get_agents_service),
):
    """Create a new turn for an agent."""
    return await svc.create_agent_turn(
        agent_id=agent_id,
        session_id=session_id,
        messages=body.messages,
        stream=body.stream,
        documents=body.documents,
        toolgroups=body.toolgroups,
        tool_config=body.tool_config,
    )


@router.post(
    "/agents/{agent_id}/session/{session_id}/turn/{turn_id}/resume",
    summary="Resume an agent turn.",
    description="Resume an agent turn with executed tool call responses.",
    deprecated=True,
)
@router_v1alpha.post(
    "/agents/{{agent_id}}/session/{{session_id}}/turn/{{turn_id}}/resume",
    summary="Resume an agent turn.",
    description="Resume an agent turn with executed tool call responses.",
)
async def resume_agent_turn(
    agent_id: Annotated[str, FastAPIPath(..., description="The ID of the agent to resume.")],
    session_id: Annotated[str, FastAPIPath(..., description="The ID of the session to resume.")],
    turn_id: Annotated[str, FastAPIPath(..., description="The ID of the turn to resume.")],
    body: AgentTurnResumeRequest = Body(...),
    svc: AgentsService = Depends(get_agents_service),
):
    """Resume an agent turn with executed tool call responses."""
    return await svc.resume_agent_turn(
        agent_id=agent_id,
        session_id=session_id,
        turn_id=turn_id,
        tool_responses=body.tool_responses,
        stream=body.stream,
    )


@router.get(
    "/agents/{agent_id}/session/{session_id}/turn/{turn_id}",
    response_model=Turn,
    summary="Retrieve an agent turn.",
    description="Retrieve an agent turn by its ID.",
    deprecated=True,
)
@router_v1alpha.get(
    "/agents/{{agent_id}}/session/{{session_id}}/turn/{{turn_id}}",
    response_model=Turn,
    summary="Retrieve an agent turn.",
    description="Retrieve an agent turn by its ID.",
)
async def get_agents_turn(
    agent_id: Annotated[str, FastAPIPath(..., description="The ID of the agent to get the turn for.")],
    session_id: Annotated[str, FastAPIPath(..., description="The ID of the session to get the turn for.")],
    turn_id: Annotated[str, FastAPIPath(..., description="The ID of the turn to get.")],
    svc: AgentsService = Depends(get_agents_service),
) -> Turn:
    """Retrieve an agent turn by its ID."""
    return await svc.get_agents_turn(agent_id=agent_id, session_id=session_id, turn_id=turn_id)


@router.get(
    "/agents/{agent_id}/session/{session_id}/turn/{turn_id}/step/{step_id}",
    response_model=AgentStepResponse,
    summary="Retrieve an agent step.",
    description="Retrieve an agent step by its ID.",
    deprecated=True,
)
@router_v1alpha.get(
    "/agents/{{agent_id}}/session/{{session_id}}/turn/{{turn_id}}/step/{{step_id}}",
    response_model=AgentStepResponse,
    summary="Retrieve an agent step.",
    description="Retrieve an agent step by its ID.",
)
async def get_agents_step(
    agent_id: Annotated[str, FastAPIPath(..., description="The ID of the agent to get the step for.")],
    session_id: Annotated[str, FastAPIPath(..., description="The ID of the session to get the step for.")],
    turn_id: Annotated[str, FastAPIPath(..., description="The ID of the turn to get the step for.")],
    step_id: Annotated[str, FastAPIPath(..., description="The ID of the step to get.")],
    svc: AgentsService = Depends(get_agents_service),
) -> AgentStepResponse:
    """Retrieve an agent step by its ID."""
    return await svc.get_agents_step(agent_id=agent_id, session_id=session_id, turn_id=turn_id, step_id=step_id)


@router.post(
    "/agents/{agent_id}/session",
    response_model=AgentSessionCreateResponse,
    summary="Create a new session for an agent.",
    description="Create a new session for an agent.",
    deprecated=True,
)
@router_v1alpha.post(
    "/agents/{{agent_id}}/session",
    response_model=AgentSessionCreateResponse,
    summary="Create a new session for an agent.",
    description="Create a new session for an agent.",
)
async def create_agent_session(
    agent_id: Annotated[str, FastAPIPath(..., description="The ID of the agent to create the session for.")],
    body: CreateAgentSessionRequest = Body(...),
    svc: AgentsService = Depends(get_agents_service),
) -> AgentSessionCreateResponse:
    """Create a new session for an agent."""
    return await svc.create_agent_session(agent_id=agent_id, session_name=body.session_name)


@router.get(
    "/agents/{agent_id}/session/{session_id}",
    response_model=Session,
    summary="Retrieve an agent session.",
    description="Retrieve an agent session by its ID.",
    deprecated=True,
)
@router_v1alpha.get(
    "/agents/{{agent_id}}/session/{{session_id}}",
    response_model=Session,
    summary="Retrieve an agent session.",
    description="Retrieve an agent session by its ID.",
)
async def get_agents_session(
    agent_id: Annotated[str, FastAPIPath(..., description="The ID of the agent to get the session for.")],
    session_id: Annotated[str, FastAPIPath(..., description="The ID of the session to get.")],
    turn_ids: list[str] | None = Query(None, description="List of turn IDs to filter the session by."),
    svc: AgentsService = Depends(get_agents_service),
) -> Session:
    """Retrieve an agent session by its ID."""
    return await svc.get_agents_session(session_id=session_id, agent_id=agent_id, turn_ids=turn_ids)


@router.delete(
    "/agents/{agent_id}/session/{session_id}",
    response_model=None,
    status_code=204,
    summary="Delete an agent session.",
    description="Delete an agent session by its ID.",
    deprecated=True,
)
@router_v1alpha.delete(
    "/agents/{{agent_id}}/session/{{session_id}}",
    response_model=None,
    status_code=204,
    summary="Delete an agent session.",
    description="Delete an agent session by its ID.",
)
async def delete_agents_session(
    agent_id: Annotated[str, FastAPIPath(..., description="The ID of the agent to delete the session for.")],
    session_id: Annotated[str, FastAPIPath(..., description="The ID of the session to delete.")],
    svc: AgentsService = Depends(get_agents_service),
) -> None:
    """Delete an agent session by its ID and its associated turns."""
    await svc.delete_agents_session(session_id=session_id, agent_id=agent_id)


@router.delete(
    "/agents/{agent_id}",
    response_model=None,
    status_code=204,
    summary="Delete an agent.",
    description="Delete an agent by its ID.",
    deprecated=True,
)
@router_v1alpha.delete(
    "/agents/{{agent_id}}",
    response_model=None,
    status_code=204,
    summary="Delete an agent.",
    description="Delete an agent by its ID.",
)
async def delete_agent(
    agent_id: Annotated[str, FastAPIPath(..., description="The ID of the agent to delete.")],
    svc: AgentsService = Depends(get_agents_service),
) -> None:
    """Delete an agent by its ID and its associated sessions and turns."""
    await svc.delete_agent(agent_id=agent_id)


@router.get(
    "/agents",
    summary="List all agents.",
    description="List all agents.",
    deprecated=True,
)
@router_v1alpha.get(
    "/agents",
    summary="List all agents.",
    description="List all agents.",
)
async def list_agents(
    start_index: int | None = Query(None, description="The index to start the pagination from."),
    limit: int | None = Query(None, description="The number of agents to return."),
    svc: AgentsService = Depends(get_agents_service),
):
    """List all agents."""
    return await svc.list_agents(start_index=start_index, limit=limit)


@router.get(
    "/agents/{agent_id}",
    response_model=Agent,
    summary="Describe an agent.",
    description="Describe an agent by its ID.",
    deprecated=True,
)
@router_v1alpha.get(
    "/agents/{{agent_id}}",
    response_model=Agent,
    summary="Describe an agent.",
    description="Describe an agent by its ID.",
)
async def get_agent(
    agent_id: Annotated[str, FastAPIPath(..., description="ID of the agent.")],
    svc: AgentsService = Depends(get_agents_service),
) -> Agent:
    """Describe an agent by its ID."""
    return await svc.get_agent(agent_id=agent_id)


@router.get(
    "/agents/{agent_id}/sessions",
    summary="List all sessions of an agent.",
    description="List all session(s) of a given agent.",
    deprecated=True,
)
@router_v1alpha.get(
    "/agents/{{agent_id}}/sessions",
    summary="List all sessions of an agent.",
    description="List all session(s) of a given agent.",
)
async def list_agent_sessions(
    agent_id: Annotated[str, FastAPIPath(..., description="The ID of the agent to list sessions for.")],
    start_index: int | None = Query(None, description="The index to start the pagination from."),
    limit: int | None = Query(None, description="The number of sessions to return."),
    svc: AgentsService = Depends(get_agents_service),
):
    """List all session(s) of a given agent."""
    return await svc.list_agent_sessions(agent_id=agent_id, start_index=start_index, limit=limit)


# OpenAI Responses API endpoints
@router.get(
    "/responses/{response_id}",
    response_model=OpenAIResponseObject,
    summary="Get a model response.",
    description="Get a model response.",
)
async def get_openai_response(
    response_id: Annotated[str, FastAPIPath(..., description="The ID of the OpenAI response to retrieve.")],
    svc: AgentsService = Depends(get_agents_service),
) -> OpenAIResponseObject:
    """Get a model response."""
    return await svc.get_openai_response(response_id=response_id)


@router.post(
    "/responses",
    summary="Create a model response.",
    description="Create a model response.",
)
async def create_openai_response(
    body: CreateOpenAIResponseRequest = Body(...),
    svc: AgentsService = Depends(get_agents_service),
):
    """Create a model response."""
    return await svc.create_openai_response(
        input=body.input,
        model=body.model,
        prompt=body.prompt,
        instructions=body.instructions,
        previous_response_id=body.previous_response_id,
        conversation=body.conversation,
        store=body.store,
        stream=body.stream,
        temperature=body.temperature,
        text=body.text,
        tools=body.tools,
        include=body.include,
        max_infer_iters=body.max_infer_iters,
        guardrails=body.guardrails,
    )


@router.get(
    "/responses",
    response_model=ListOpenAIResponseObject,
    summary="List all responses.",
    description="List all responses.",
)
async def list_openai_responses(
    after: str | None = Query(None, description="The ID of the last response to return."),
    limit: int | None = Query(50, description="The number of responses to return."),
    model: str | None = Query(None, description="The model to filter responses by."),
    order: Order | None = Query(
        Order.desc, description="The order to sort responses by when sorted by created_at ('asc' or 'desc')."
    ),
    svc: AgentsService = Depends(get_agents_service),
) -> ListOpenAIResponseObject:
    """List all responses."""
    return await svc.list_openai_responses(after=after, limit=limit, model=model, order=order)


@router.get(
    "/responses/{response_id}/input_items",
    response_model=ListOpenAIResponseInputItem,
    summary="List input items.",
    description="List input items.",
)
async def list_openai_response_input_items(
    response_id: Annotated[str, FastAPIPath(..., description="The ID of the response to retrieve input items for.")],
    after: str | None = Query(None, description="An item ID to list items after, used for pagination."),
    before: str | None = Query(None, description="An item ID to list items before, used for pagination."),
    include: list[str] | None = Query(None, description="Additional fields to include in the response."),
    limit: int | None = Query(
        20,
        description="A limit on the number of objects to be returned. Limit can range between 1 and 100, and the default is 20.",
        ge=1,
        le=100,
    ),
    order: Order | None = Query(Order.desc, description="The order to return the input items in. Default is desc."),
    svc: AgentsService = Depends(get_agents_service),
) -> ListOpenAIResponseInputItem:
    """List input items."""
    return await svc.list_openai_response_input_items(
        response_id=response_id, after=after, before=before, include=include, limit=limit, order=order
    )


@router.delete(
    "/responses/{response_id}",
    response_model=OpenAIDeleteResponseObject,
    summary="Delete a response.",
    description="Delete a response.",
)
async def delete_openai_response(
    response_id: Annotated[str, FastAPIPath(..., description="The ID of the OpenAI response to delete.")],
    svc: AgentsService = Depends(get_agents_service),
) -> OpenAIDeleteResponseObject:
    """Delete a response."""
    return await svc.delete_openai_response(response_id=response_id)


# For backward compatibility with the router registry system
def create_agents_router(impl_getter) -> APIRouter:
    """Create a FastAPI router for the Agents API (legacy compatibility)."""
    main_router = APIRouter()
    main_router.include_router(router)
    main_router.include_router(router_v1alpha)
    return main_router


# Register the router factory
register_router(Api.agents, create_agents_router)

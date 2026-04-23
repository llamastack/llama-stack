# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""FastAPI router for the Anthropic Agents API.

This module defines the FastAPI router for the /v1alpha/agents endpoint,
serving the Anthropic Managed Agents API format.
"""

import logging  # allow-direct-logging
from typing import Annotated

from fastapi import APIRouter, Body, Path, Query
from fastapi.responses import JSONResponse

from llama_stack_api.common.errors import AgentNotFoundError
from llama_stack_api.router_utils import standard_responses
from llama_stack_api.version import LLAMA_STACK_API_V1ALPHA

from .api import Agents
from .models import (
    AgentObject,
    AnthropicAgentsErrorResponse,
    ArchiveAgentRequest,
    CreateAgentRequest,
    ListAgentsRequest,
    ListAgentsResponse,
    UpdateAgentRequest,
    _AnthropicErrorDetail,
)

logger = logging.LoggerAdapter(logging.getLogger(__name__), {"category": "agents"})


def _agents_error_response(status_code: int, message: str, error_type: str = "error") -> JSONResponse:
    """Create an Anthropic-format error JSONResponse."""
    body = AnthropicAgentsErrorResponse(
        error=_AnthropicErrorDetail(type=error_type, message=message),
    )
    return JSONResponse(status_code=status_code, content=body.model_dump())


def create_router(impl: Agents) -> APIRouter:
    """Create a FastAPI router for the Anthropic Agents API.

    Args:
        impl: The Agents implementation instance

    Returns:
        APIRouter configured for the Agents API
    """
    router = APIRouter(
        prefix=f"/{LLAMA_STACK_API_V1ALPHA}",
        tags=["Agents (v1alpha)"],
        responses=standard_responses,
    )

    @router.post(
        "/agents",
        summary="Create an agent configuration",
        description="Create a new agent with model, tools, and system prompt configuration.",
        status_code=200,
        response_model=AgentObject,
    )
    async def create_agent(
        params: Annotated[CreateAgentRequest, Body(...)],
    ) -> JSONResponse:
        try:
            result = await impl.create_agent(params)
            return JSONResponse(content=result.model_dump(mode="json"))
        except NotImplementedError as e:
            return _agents_error_response(501, str(e), "not_implemented")
        except ValueError as e:
            return _agents_error_response(400, str(e), "invalid_request")
        except Exception:
            logger.exception("Failed to create agent")
            return _agents_error_response(500, "Internal server error", "internal_error")

    @router.get(
        "/agents",
        summary="List agents",
        description="List all agent configurations with optional filtering and pagination.",
        status_code=200,
        response_model=ListAgentsResponse,
    )
    async def list_agents(
        limit: Annotated[int | None, Query(ge=1, le=100)] = None,
        after: Annotated[str | None, Query()] = None,
        archived: Annotated[bool | None, Query()] = None,
    ) -> JSONResponse:
        try:
            request = ListAgentsRequest(limit=limit, after=after, archived=archived)
            result = await impl.list_agents(request)
            return JSONResponse(content=result.model_dump(mode="json"))
        except NotImplementedError as e:
            return _agents_error_response(501, str(e), "not_implemented")
        except Exception:
            logger.exception("Failed to list agents")
            return _agents_error_response(500, "Internal server error", "internal_error")

    @router.get(
        "/agents/{agent_id}",
        summary="Retrieve an agent",
        description="Get an agent configuration by ID.",
        status_code=200,
        response_model=AgentObject,
    )
    async def get_agent(
        agent_id: Annotated[str, Path(description="Agent ID")],
    ) -> JSONResponse:
        try:
            result = await impl.get_agent(agent_id)
            return JSONResponse(content=result.model_dump(mode="json"))
        except AgentNotFoundError as e:
            return _agents_error_response(404, str(e), "not_found")
        except NotImplementedError as e:
            return _agents_error_response(501, str(e), "not_implemented")
        except Exception:
            logger.exception("Failed to get agent", agent_id=agent_id)
            return _agents_error_response(500, "Internal server error", "internal_error")

    @router.post(
        "/agents/{agent_id}",
        summary="Update an agent (creates new version)",
        description="Update an agent configuration. Creates a new version while preserving previous versions.",
        status_code=200,
        response_model=AgentObject,
    )
    async def update_agent(
        agent_id: Annotated[str, Path(description="Agent ID")],
        params: Annotated[UpdateAgentRequest, Body(...)],
    ) -> JSONResponse:
        try:
            result = await impl.update_agent(agent_id, params)
            return JSONResponse(content=result.model_dump(mode="json"))
        except AgentNotFoundError as e:
            return _agents_error_response(404, str(e), "not_found")
        except NotImplementedError as e:
            return _agents_error_response(501, str(e), "not_implemented")
        except ValueError as e:
            return _agents_error_response(400, str(e), "invalid_request")
        except Exception:
            logger.exception("Failed to update agent", agent_id=agent_id)
            return _agents_error_response(500, "Internal server error", "internal_error")

    @router.post(
        "/agents/{agent_id}/archive",
        summary="Archive an agent",
        description="Archive an agent configuration. Archived agents cannot be used for new sessions.",
        status_code=200,
        response_model=AgentObject,
    )
    async def archive_agent(
        agent_id: Annotated[str, Path(description="Agent ID")],
        params: Annotated[ArchiveAgentRequest | None, Body()] = None,
    ) -> JSONResponse:
        try:
            result = await impl.archive_agent(agent_id, params or ArchiveAgentRequest())
            return JSONResponse(content=result.model_dump(mode="json"))
        except AgentNotFoundError as e:
            return _agents_error_response(404, str(e), "not_found")
        except NotImplementedError as e:
            return _agents_error_response(501, str(e), "not_implemented")
        except Exception:
            logger.exception("Failed to archive agent", agent_id=agent_id)
            return _agents_error_response(500, "Internal server error", "internal_error")

    return router

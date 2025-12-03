# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
FastAPI app creation for OpenAPI generation.
"""

import inspect
from typing import Any

from fastapi import FastAPI

from llama_stack.core.resolver import api_protocol_map
from llama_stack.core.server.fastapi_router_registry import build_fastapi_router, has_router
from llama_stack_api import Api

from .state import _protocol_methods_cache


def _get_protocol_method(api: Api, method_name: str) -> Any | None:
    """
    Get a protocol method function by API and method name.
    Uses caching to avoid repeated lookups.

    Args:
        api: The API enum
        method_name: The method name (function name)

    Returns:
        The function object, or None if not found
    """
    global _protocol_methods_cache

    if _protocol_methods_cache is None:
        _protocol_methods_cache = {}
        protocols = api_protocol_map()
        from llama_stack_api.tools import SpecialToolGroup, ToolRuntime

        toolgroup_protocols = {
            SpecialToolGroup.rag_tool: ToolRuntime,
        }

        for api_key, protocol in protocols.items():
            method_map: dict[str, Any] = {}
            protocol_methods = inspect.getmembers(protocol, predicate=inspect.isfunction)
            for name, method in protocol_methods:
                method_map[name] = method

            # Handle tool_runtime special case
            if api_key == Api.tool_runtime:
                for tool_group, sub_protocol in toolgroup_protocols.items():
                    sub_protocol_methods = inspect.getmembers(sub_protocol, predicate=inspect.isfunction)
                    for name, method in sub_protocol_methods:
                        if hasattr(method, "__webmethod__"):
                            method_map[f"{tool_group.value}.{name}"] = method

            _protocol_methods_cache[api_key] = method_map

    return _protocol_methods_cache.get(api, {}).get(method_name)


def create_llama_stack_app() -> FastAPI:
    """
    Create a FastAPI app that represents the Llama Stack API.
    This uses both router-based routes (for migrated APIs) and the existing
    route discovery system for legacy webmethod-based routes.
    """
    app = FastAPI(
        title="Llama Stack API",
        description="A comprehensive API for building and deploying AI applications",
        version="1.0.0",
        servers=[
            {"url": "http://any-hosted-llama-stack.com"},
        ],
    )

    # Include routers for APIs that have them
    protocols = api_protocol_map()
    for api in protocols.keys():
        # For OpenAPI generation, we don't need a real implementation
        if not has_router(api):
            continue
        app.include_router(build_fastapi_router(api, None))

    # Get all API routes (for legacy webmethod-based routes)
    from llama_stack.core.server.routes import get_all_api_routes

    api_routes = get_all_api_routes()

    # Create FastAPI routes from the discovered routes (skip APIs that have routers)
    from . import endpoints

    for api, routes in api_routes.items():
        # Skip APIs that have routers - they're already included above
        if has_router(api):
            continue

        for route, webmethod in routes:
            # Convert the route to a FastAPI endpoint
            endpoints._create_fastapi_endpoint(app, route, webmethod, api)

    return app

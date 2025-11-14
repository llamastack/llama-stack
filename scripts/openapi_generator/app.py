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
    This uses the existing route discovery system to automatically find all routes.
    """
    app = FastAPI(
        title="Llama Stack API",
        description="A comprehensive API for building and deploying AI applications",
        version="1.0.0",
        servers=[
            {"url": "http://any-hosted-llama-stack.com"},
        ],
    )

    # Get all API routes
    from llama_stack.core.server.routes import get_all_api_routes

    api_routes = get_all_api_routes()

    # Create FastAPI routes from the discovered routes
    from . import endpoints

    for api, routes in api_routes.items():
        for route, webmethod in routes:
            # Convert the route to a FastAPI endpoint
            endpoints._create_fastapi_endpoint(app, route, webmethod, api)

    return app

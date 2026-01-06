# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Router utilities for FastAPI routers.

This module provides utilities to create FastAPI routers from API packages.
APIs with routers are explicitly listed here.
"""

from collections.abc import Callable
from typing import Any, cast

from fastapi import APIRouter
from fastapi.routing import APIRoute

from llama_stack_api import admin, batches, benchmarks, datasets, files, inspect_api, providers

# Router factories for APIs that have FastAPI routers
# Add new APIs here as they are migrated to the router system
from llama_stack_api.datatypes import Api

_ROUTER_FACTORIES: dict[str, Callable[[Any], APIRouter]] = {
    "admin": admin.fastapi_routes.create_router,
    "batches": batches.fastapi_routes.create_router,
    "benchmarks": benchmarks.fastapi_routes.create_router,
    "datasets": datasets.fastapi_routes.create_router,
    "providers": providers.fastapi_routes.create_router,
    "inspect": inspect_api.fastapi_routes.create_router,
    "files": files.fastapi_routes.create_router,
}


def has_router(api: "Api") -> bool:
    """Check if an API has a router factory.

    Args:
        api: The API enum value

    Returns:
        True if the API has a router factory, False otherwise
    """
    return api.value in _ROUTER_FACTORIES


def build_fastapi_router(api: "Api", impl: Any) -> APIRouter | None:
    """Build a router for an API by combining its router factory with the implementation.

    Args:
        api: The API enum value
        impl: The implementation instance for the API

    Returns:
        APIRouter if the API has a router factory, None otherwise
    """
    router_factory = _ROUTER_FACTORIES.get(api.value)
    if router_factory is None:
        return None

    # cast is safe here: all router factories in API packages are required to return APIRouter.
    # If a router factory returns the wrong type, it will fail at runtime when
    # app.include_router(router) is called
    return cast(APIRouter, router_factory(impl))


def get_router_routes(router: APIRouter) -> list[APIRoute]:
    """Extract APIRoute objects from a FastAPI router.

    Args:
        router: The FastAPI router to extract routes from

    Returns:
        List of APIRoute objects from the router (preserves tags and other metadata)
    """
    routes = []

    for route in router.routes:
        # FastAPI routers use APIRoute objects, which have path, methods, tags, etc.
        if isinstance(route, APIRoute):
            routes.append(route)

    return routes

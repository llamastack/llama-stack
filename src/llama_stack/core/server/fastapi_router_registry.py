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
from starlette.routing import Route

from llama_stack_api import batches, benchmarks, datasets

# Router factories for APIs that have FastAPI routers
# Add new APIs here as they are migrated to the router system
from llama_stack_api.datatypes import Api

_ROUTER_FACTORIES: dict[str, Callable[[Any], APIRouter]] = {
    "batches": batches.fastapi_routes.create_router,
    "benchmarks": benchmarks.fastapi_routes.create_router,
    "datasets": datasets.fastapi_routes.create_router,
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


def get_router_routes(router: APIRouter) -> list[Route]:
    """Extract routes from a FastAPI router.

    Args:
        router: The FastAPI router to extract routes from

    Returns:
        List of Route objects from the router
    """
    routes = []

    for route in router.routes:
        # FastAPI routers use APIRoute objects, which have path and methods attributes
        if isinstance(route, APIRoute):
            # Combine router prefix with route path
            routes.append(
                Route(
                    path=route.path,
                    methods=route.methods,
                    name=route.name,
                    endpoint=route.endpoint,
                )
            )
        elif isinstance(route, Route):
            # Fallback for regular Starlette Route objects
            routes.append(
                Route(
                    path=route.path,
                    methods=route.methods,
                    name=route.name,
                    endpoint=route.endpoint,
                )
            )

    return routes

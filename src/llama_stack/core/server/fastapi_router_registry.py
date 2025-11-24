# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Router utilities for FastAPI routers.

This module provides utilities to create FastAPI routers from API packages.
APIs with routers are explicitly listed here.
"""

from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter

if TYPE_CHECKING:
    from llama_stack_api.datatypes import Api

# Router factories for APIs that have FastAPI routers
# Add new APIs here as they are migrated to the router system
from llama_stack_api.batches.fastapi_routes import create_router as create_batches_router

_ROUTER_FACTORIES: dict[str, APIRouter] = {
    "batches": create_batches_router,
}


def build_router(api: "Api", impl: Any) -> APIRouter | None:
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

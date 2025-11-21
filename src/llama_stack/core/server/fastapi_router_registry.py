# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Router utilities for FastAPI routers.

This module provides utilities to discover and create FastAPI routers from API packages.
Routers are automatically discovered by checking for routes modules in each API package.
"""

import importlib
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter

if TYPE_CHECKING:
    from llama_stack_api.datatypes import Api


def has_router(api: "Api") -> bool:
    """Check if an API has a router factory in its routes module.

    Args:
        api: The API enum value

    Returns:
        True if the API has a routes module with a create_router function
    """
    try:
        routes_module = importlib.import_module(f"llama_stack_api.{api.value}.fastapi_routes")
        return hasattr(routes_module, "create_router")
    except (ImportError, AttributeError):
        return False


def build_router(api: "Api", impl: Any) -> APIRouter | None:
    """Build a router for an API by combining its router factory with the implementation.

    This function discovers the router factory from the API package's routes module
    and calls it with the implementation to create the final router instance.

    Args:
        api: The API enum value
        impl: The implementation instance for the API

    Returns:
        APIRouter if the API has a routes module with create_router, None otherwise
    """
    try:
        routes_module = importlib.import_module(f"llama_stack_api.{api.value}.fastapi_routes")
        if hasattr(routes_module, "create_router"):
            router_factory = routes_module.create_router
            return router_factory(impl)
    except (ImportError, AttributeError):
        pass

    return None

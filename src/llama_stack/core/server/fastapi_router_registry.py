# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Router utilities for FastAPI routers.

This module auto-discovers FastAPI routers from llama_stack_api packages.
Each API package that has a `fastapi_routes` submodule with a `create_router`
factory is automatically registered.

External APIs can also provide a `create_router` function in their module
(the same module that provides `available_providers`).
"""

import importlib
from collections.abc import Callable
from typing import Any, cast

from fastapi import APIRouter
from fastapi.routing import APIRoute

from llama_stack_api.datatypes import Api, ExternalApiSpec

# Api enum values that don't match their package name in llama_stack_api
_API_TO_PACKAGE: dict[str, str] = {
    "inspect": "inspect_api",
    "tool_groups": "tools",
}


def _discover_router_factories() -> dict[str, Callable[[Any], APIRouter]]:
    """Auto-discover router factories from llama_stack_api packages.

    For each Api enum value, try to import
    `llama_stack_api.<package>.fastapi_routes.create_router`.
    APIs without a fastapi_routes module (e.g. vector_stores, tool_runtime)
    are silently skipped.
    """
    factories: dict[str, Callable[[Any], APIRouter]] = {}
    for api in Api:
        package_name = _API_TO_PACKAGE.get(api.value, api.value)
        module_path = f"llama_stack_api.{package_name}.fastapi_routes"
        try:
            module = importlib.import_module(module_path)
            create_router = getattr(module, "create_router", None)
            if create_router is not None:
                factories[api.value] = create_router
        except (ImportError, ModuleNotFoundError):
            pass
    return factories


_ROUTER_FACTORIES: dict[str, Callable[[Any], APIRouter]] = _discover_router_factories()


def register_external_api_routers(external_apis: dict[Api, ExternalApiSpec]) -> None:
    """Register router factories from external API modules.

    External APIs can provide a `create_router(impl) -> APIRouter` function
    in their module to define FastAPI routes.
    """
    for api, api_spec in external_apis.items():
        if api.value in _ROUTER_FACTORIES:
            continue
        try:
            module = importlib.import_module(api_spec.module)
            create_router = getattr(module, "create_router", None)
            if create_router is not None:
                _ROUTER_FACTORIES[api.value] = create_router
        except (ImportError, ModuleNotFoundError):
            pass


def build_fastapi_router(api: "Api", impl: Any) -> APIRouter | None:
    """Build a router for an API using its auto-discovered router factory."""
    router_factory = _ROUTER_FACTORIES.get(api.value)
    if router_factory is None:
        return None

    return cast(APIRouter, router_factory(impl))


def get_router_routes(router: APIRouter) -> list[APIRoute]:
    """Extract APIRoute objects from a FastAPI router."""
    return [route for route in router.routes if isinstance(route, APIRoute)]

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Router utilities for FastAPI routers.

This module registers FastAPI routers from llama_stack_api packages.
Each API that has a `fastapi_routes` submodule with a `create_router`
factory is explicitly listed here.

External APIs can also provide a `create_router` function in their module
(the same module that provides `available_providers`).
"""

import importlib
from collections.abc import Callable
from typing import Any, cast

from fastapi import APIRouter
from fastapi.routing import APIRoute

from llama_stack_api import (
    admin,
    batches,
    benchmarks,
    connectors,
    conversations,
    datasetio,
    datasets,
    eval,
    file_processors,
    files,
    inference,
    inspect_api,
    models,
    prompts,
    providers,
    responses,
    safety,
    scoring,
    scoring_functions,
    shields,
    tools,
    vector_io,
)
from llama_stack_api.datatypes import Api, ExternalApiSpec

# Router factories for APIs that have FastAPI routers
# Add new APIs here as they are migrated to the router system
_ROUTER_FACTORIES: dict[str, Callable[[Any], APIRouter]] = {
    "admin": admin.fastapi_routes.create_router,
    "responses": responses.fastapi_routes.create_router,
    "batches": batches.fastapi_routes.create_router,
    "benchmarks": benchmarks.fastapi_routes.create_router,
    "connectors": connectors.fastapi_routes.create_router,
    "conversations": conversations.fastapi_routes.create_router,
    "datasetio": datasetio.fastapi_routes.create_router,
    "datasets": datasets.fastapi_routes.create_router,
    "eval": eval.fastapi_routes.create_router,
    "file_processors": file_processors.fastapi_routes.create_router,
    "files": files.fastapi_routes.create_router,
    "inference": inference.fastapi_routes.create_router,
    "inspect": inspect_api.fastapi_routes.create_router,
    "models": models.fastapi_routes.create_router,
    "prompts": prompts.fastapi_routes.create_router,
    "providers": providers.fastapi_routes.create_router,
    "safety": safety.fastapi_routes.create_router,
    "scoring": scoring.fastapi_routes.create_router,
    "scoring_functions": scoring_functions.fastapi_routes.create_router,
    "shields": shields.fastapi_routes.create_router,
    "tool_groups": tools.fastapi_routes.create_router,
    "vector_io": vector_io.fastapi_routes.create_router,
}


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

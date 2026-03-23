# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import inspect
import re
from collections.abc import Callable
from typing import Any

from aiohttp import hdrs
from starlette.routing import Route

from llama_stack.core.resolver import api_protocol_map
from llama_stack.core.server.fastapi_router_registry import (
    _ROUTER_FACTORIES,
    build_fastapi_router,
    get_router_routes,
)
from llama_stack_api import Api, ExternalApiSpec, WebMethod
from llama_stack_api.router_utils import PUBLIC_ROUTE_KEY

EndpointFunc = Callable[..., Any]
PathParams = dict[str, str]
RouteInfo = tuple[EndpointFunc, str, WebMethod]
PathImpl = dict[str, RouteInfo]
RouteImpls = dict[str, PathImpl]
RouteMatch = tuple[EndpointFunc, PathParams, str, WebMethod]


def _convert_path_to_regex(path: str) -> str:
    # Convert {param} to named capture groups
    # handle {param:path} as well which allows for forward slashes in the param value
    pattern = re.sub(
        r"{(\w+)(?::path)?}",
        lambda m: f"(?P<{m.group(1)}>{'[^/]+' if not m.group(0).endswith(':path') else '.+'})",
        path,
    )

    return f"^{pattern}$"


def get_all_api_routes(
    external_apis: dict[Api, ExternalApiSpec] | None = None,
) -> dict[Api, list[tuple[Route, WebMethod]]]:
    """Get all API routes from webmethod-based protocols.

    Used for external APIs that define routes via @webmethod decorators
    rather than FastAPI routers.
    """
    apis = {}

    protocols = api_protocol_map(external_apis)
    for api, protocol in protocols.items():
        routes = []
        protocol_methods = inspect.getmembers(protocol, predicate=inspect.isfunction)

        for name, method in protocol_methods:
            webmethods = getattr(method, "__webmethods__", [])
            if not webmethods:
                continue

            for webmethod in webmethods:
                path = f"/{webmethod.level}/{webmethod.route.lstrip('/')}"
                if webmethod.method == hdrs.METH_GET:
                    http_method = hdrs.METH_GET
                elif webmethod.method == hdrs.METH_DELETE:
                    http_method = hdrs.METH_DELETE
                else:
                    http_method = hdrs.METH_POST
                routes.append(
                    (Route(path=path, methods=[http_method], name=name, endpoint=None), webmethod)  # type: ignore[arg-type]
                )

        apis[api] = routes

    return apis


def initialize_route_impls(impls, external_apis: dict[Api, ExternalApiSpec] | None = None) -> RouteImpls:
    api_to_routes = get_all_api_routes(external_apis)
    route_impls: RouteImpls = {}

    for api_name in _ROUTER_FACTORIES.keys():
        api = Api(api_name)
        if api not in impls:
            continue
        impl = impls[api]
        router = build_fastapi_router(api, impl)
        if router:
            router_routes = get_router_routes(router)
            for route in router_routes:
                func = route.endpoint
                if func is None:
                    continue

                available_methods = [m for m in (route.methods or []) if m != "HEAD"]
                if not available_methods:
                    continue
                method = available_methods[0].lower()

                if method not in route_impls:
                    route_impls[method] = {}

                # Routes with openapi_extra[PUBLIC_ROUTE_KEY]=True don't require authentication
                is_public = (route.openapi_extra or {}).get(PUBLIC_ROUTE_KEY, False)
                webmethod = WebMethod(
                    descriptive_name=None,
                    require_authentication=not is_public,
                )
                route_impls[method][_convert_path_to_regex(route.path)] = (
                    func,
                    route.path,
                    webmethod,
                )

    # Process routes from external APIs using @webmethod decorators
    for api, api_routes in api_to_routes.items():
        if api.value in _ROUTER_FACTORIES:
            continue
        if api not in impls:
            continue
        for legacy_route, webmethod in api_routes:
            impl = impls[api]
            func = getattr(impl, legacy_route.name)
            available_methods = [m for m in (legacy_route.methods or []) if m != "HEAD"]
            if not available_methods:
                continue
            method = available_methods[0].lower()
            if method not in route_impls:
                route_impls[method] = {}
            route_impls[method][_convert_path_to_regex(legacy_route.path)] = (
                func,
                legacy_route.path,
                webmethod,
            )

    return route_impls


def find_matching_route(method: str, path: str, route_impls: RouteImpls) -> RouteMatch:
    """Find the matching endpoint implementation for a given method and path.

    Returns a tuple of (endpoint_function, path_params, route_path, webmethod_metadata).

    Raises ValueError if no matching endpoint is found.
    """
    impls = route_impls.get(method.lower())
    if not impls:
        raise ValueError(f"No endpoint found for {path}")

    for regex, (func, route_path, webmethod) in impls.items():
        match = re.match(regex, path)
        if match:
            path_params = match.groupdict()
            return func, path_params, route_path, webmethod

    raise ValueError(f"No endpoint found for {path}")

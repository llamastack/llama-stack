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

from llama_stack.apis.datatypes import Api, ExternalApiSpec
from llama_stack.core.resolver import api_protocol_map
from llama_stack.schema_utils import WebMethod

EndpointFunc = Callable[..., Any]
PathParams = dict[str, str]
RouteInfo = tuple[EndpointFunc, str, WebMethod]
PathImpl = dict[str, RouteInfo]
RouteImpls = dict[str, PathImpl]
RouteMatch = tuple[EndpointFunc, PathParams, str, WebMethod]


def get_all_api_routes(
    external_apis: dict[Api, ExternalApiSpec] | None = None,
) -> dict[Api, list[tuple[Route, WebMethod]]]:
    apis = {}

    protocols = api_protocol_map(external_apis)
    for api, protocol in protocols.items():
        routes = []
        protocol_methods = inspect.getmembers(protocol, predicate=inspect.isfunction)

        for name, method in protocol_methods:
            # Get all webmethods for this method (supports multiple decorators)
            webmethods = getattr(method, "__webmethods__", [])
            if not webmethods:
                continue

            # Create routes for each webmethod decorator
            for webmethod in webmethods:
                path = f"/{webmethod.level}/{webmethod.route.lstrip('/')}"
                if webmethod.method == hdrs.METH_GET:
                    http_method = hdrs.METH_GET
                elif webmethod.method == hdrs.METH_DELETE:
                    http_method = hdrs.METH_DELETE
                else:
                    http_method = hdrs.METH_POST
                routes.append(
                    # setting endpoint to None since don't use a Router object
                    (Route(path=path, methods=[http_method], name=name, endpoint=None), webmethod)  # type: ignore[arg-type]
                )

        apis[api] = routes

    return apis


def initialize_route_impls(impls, external_apis: dict[Api, ExternalApiSpec] | None = None) -> RouteImpls:
    api_to_routes = get_all_api_routes(external_apis)
    route_impls: RouteImpls = {}

    def _convert_path_to_regex(path: str) -> str:
        # Convert {param} to named capture groups
        # handle {param:path} as well which allows for forward slashes in the param value
        pattern = re.sub(
            r"{(\w+)(?::path)?}",
            lambda m: f"(?P<{m.group(1)}>{'[^/]+' if not m.group(0).endswith(':path') else '.+'})",
            path,
        )

        return f"^{pattern}$"

    for api, api_routes in api_to_routes.items():
        if api not in impls:
            continue
        for route, webmethod in api_routes:
            impl = impls[api]
            func = getattr(impl, route.name)
            # Get the first (and typically only) method from the set, filtering out HEAD
            available_methods = [m for m in (route.methods or []) if m != "HEAD"]
            if not available_methods:
                continue  # Skip if only HEAD method is available
            method = available_methods[0].lower()
            if method not in route_impls:
                route_impls[method] = {}
            route_impls[method][_convert_path_to_regex(route.path)] = (
                func,
                route.path,
                webmethod,
            )

    return route_impls


def find_matching_route(method: str, path: str, route_impls: RouteImpls) -> RouteMatch:
    """Find the matching endpoint implementation for a given method and path.

    Args:
        method: HTTP method (GET, POST, etc.)
        path: URL path to match against
        route_impls: A dictionary of endpoint implementations

    Returns:
        A tuple of (endpoint_function, path_params, route_path, webmethod_metadata)

    Raises:
        ValueError: If no matching endpoint is found
    """
    impls = route_impls.get(method.lower())
    if not impls:
        raise ValueError(f"No endpoint found for {path}")

    for regex, (func, route_path, webmethod) in impls.items():
        match = re.match(regex, path)
        if match:
            # Extract named groups from the regex match
            path_params = match.groupdict()
            return func, path_params, route_path, webmethod

    raise ValueError(f"No endpoint found for {path}")

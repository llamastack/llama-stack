# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from aiohttp import hdrs

from llama_stack.core.external import ExternalApiSpec
from llama_stack.core.server.fastapi_router_registry import has_router
from llama_stack.core.server.routes import find_matching_route, initialize_route_impls
from llama_stack.core.telemetry.tracing import end_trace, start_trace
from llama_stack.log import get_logger
from llama_stack_api.datatypes import Api
from llama_stack_api.version import (
    LLAMA_STACK_API_V1,
    LLAMA_STACK_API_V1ALPHA,
    LLAMA_STACK_API_V1BETA,
)

logger = get_logger(name=__name__, category="core::server")

# Valid API version levels - all routes must start with one of these
VALID_API_LEVELS = {LLAMA_STACK_API_V1, LLAMA_STACK_API_V1ALPHA, LLAMA_STACK_API_V1BETA}


class TracingMiddleware:
    def __init__(self, app, impls, external_apis: dict[str, ExternalApiSpec]):
        self.app = app
        self.impls = impls
        self.external_apis = external_apis
        # FastAPI built-in paths that should bypass custom routing
        self.fastapi_paths = ("/docs", "/redoc", "/openapi.json", "/favicon.ico", "/static")

    def _is_router_based_route(self, path: str) -> bool:
        """Check if a path belongs to a router-based API.

        Router-based APIs use FastAPI routers instead of the old webmethod system.
        We need to check if the path matches any router-based API prefix.
        """
        # Extract API name from path (e.g., /v1/batches -> batches)
        # Paths must start with a valid API level: /v1/{api_name} or /v1alpha/{api_name} or /v1beta/{api_name}
        parts = path.strip("/").split("/")
        if len(parts) >= 2 and parts[0] in VALID_API_LEVELS:
            api_name = parts[1]
            try:
                api = Api(api_name)
                return has_router(api)
            except (ValueError, KeyError):
                # Not a known API or not router-based
                return False
        return False

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "lifespan":
            return await self.app(scope, receive, send)

        path = scope.get("path", "")

        # Check if the path is a FastAPI built-in path
        if path.startswith(self.fastapi_paths):
            # Pass through to FastAPI's built-in handlers
            logger.debug(f"Bypassing custom routing for FastAPI built-in path: {path}")
            return await self.app(scope, receive, send)

        # Check if this is a router-based route - if so, pass through to FastAPI
        # Router-based routes are handled by FastAPI directly, so we skip the old route lookup
        # but still need to set up tracing
        is_router_based = self._is_router_based_route(path)
        if is_router_based:
            logger.debug(f"Router-based route detected: {path}, setting up tracing")
            # Set up tracing for router-based routes
            trace_attributes = {"__location__": "server", "raw_path": path}

            # Extract W3C trace context headers and store as trace attributes
            headers = dict(scope.get("headers", []))
            traceparent = headers.get(b"traceparent", b"").decode()
            if traceparent:
                trace_attributes["traceparent"] = traceparent
            tracestate = headers.get(b"tracestate", b"").decode()
            if tracestate:
                trace_attributes["tracestate"] = tracestate

            trace_context = await start_trace(path, trace_attributes)

            async def send_with_trace_id(message):
                if message["type"] == "http.response.start":
                    headers = message.get("headers", [])
                    headers.append([b"x-trace-id", str(trace_context.trace_id).encode()])
                    message["headers"] = headers
                await send(message)

            try:
                return await self.app(scope, receive, send_with_trace_id)
            finally:
                # Always end trace, even if exception occurred
                # FastAPI's exception handler will handle the exception and send the response
                # The exception will continue to propagate for logging, which is normal
                try:
                    await end_trace()
                except Exception:
                    logger.exception("Error ending trace")

        if not hasattr(self, "route_impls"):
            self.route_impls = initialize_route_impls(self.impls, self.external_apis)

        try:
            _, _, route_path, webmethod = find_matching_route(
                scope.get("method", hdrs.METH_GET), path, self.route_impls
            )
        except ValueError:
            # If no matching endpoint is found, pass through to FastAPI
            logger.debug(f"No matching route found for path: {path}, falling back to FastAPI")
            return await self.app(scope, receive, send)

        # Log deprecation warning if route is deprecated
        if getattr(webmethod, "deprecated", False):
            logger.warning(
                f"DEPRECATED ROUTE USED: {scope.get('method', 'GET')} {path} - "
                f"This route is deprecated and may be removed in a future version. "
                f"Please check the docs for the supported version."
            )

        trace_attributes = {"__location__": "server", "raw_path": path}

        # Extract W3C trace context headers and store as trace attributes
        headers = dict(scope.get("headers", []))
        traceparent = headers.get(b"traceparent", b"").decode()
        if traceparent:
            trace_attributes["traceparent"] = traceparent
        tracestate = headers.get(b"tracestate", b"").decode()
        if tracestate:
            trace_attributes["tracestate"] = tracestate

        trace_path = webmethod.descriptive_name or route_path
        trace_context = await start_trace(trace_path, trace_attributes)

        async def send_with_trace_id(message):
            if message["type"] == "http.response.start":
                headers = message.get("headers", [])
                headers.append([b"x-trace-id", str(trace_context.trace_id).encode()])
                message["headers"] = headers
            await send(message)

        try:
            return await self.app(scope, receive, send_with_trace_id)
        finally:
            await end_trace()

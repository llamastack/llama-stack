# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import re

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from llama_stack.log import get_logger

logger = get_logger(name=__name__, category="core::middleware")

# Patterns for endpoints that need query parameter injection
QUERY_PARAM_ENDPOINTS = [
    # /vector_stores/{vector_store_id}/files/{file_id}/content
    re.compile(r"/vector_stores/[^/]+/files/[^/]+/content$"),
]


class QueryParamsMiddleware(BaseHTTPMiddleware):
    """Middleware to inject query parameters into extra_query for specific endpoints"""

    async def dispatch(self, request: Request, call_next):
        # Check if this is an endpoint that needs query parameter injection
        if request.method == "GET" and any(pattern.search(str(request.url.path)) for pattern in QUERY_PARAM_ENDPOINTS):
            # Extract all query parameters and convert to appropriate types
            extra_query = {}
            query_params = dict(request.query_params)

            # Convert query parameters using JSON parsing for robust type conversion
            for key, value in query_params.items():
                try:
                    # parse as JSON to handles booleans, numbers, strings properly
                    extra_query[key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    # if parsing fails, keep as string
                    extra_query[key] = value

            if extra_query:
                # Store the extra_query in request state so we can access it later
                request.state.extra_query = extra_query
                logger.debug(f"QueryParamsMiddleware extracted extra_query: {extra_query}")

        response = await call_next(request)
        return response

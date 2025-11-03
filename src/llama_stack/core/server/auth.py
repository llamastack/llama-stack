# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json

import httpx

from llama_stack.core.datatypes import AuthenticationConfig
from llama_stack.core.server.auth_providers import create_auth_provider
from llama_stack.log import get_logger

logger = get_logger(name=__name__, category="core::auth")


class AuthenticationMiddleware:
    """Middleware that authenticates requests using configured authentication provider.

    This middleware:
    1. Extracts the Bearer token from the Authorization header
    2. Uses the configured auth provider to validate the token
    3. Extracts user attributes from the provider's response
    4. Makes these attributes available to the route handlers for access control

    Unauthenticated Access:
    Public endpoints (like /health and /version) are configured in the middleware
    and bypass authentication. All other endpoints require authentication.

    The middleware supports multiple authentication providers through the AuthProvider interface:
    - Kubernetes: Validates tokens against the Kubernetes API server
    - Custom: Validates tokens against a custom endpoint

    Authentication Request Format for Custom Auth Provider:
    ```json
    {
        "api_key": "the-api-key-extracted-from-auth-header",
        "request": {
            "path": "/models/list",
            "headers": {
                "content-type": "application/json",
                "user-agent": "..."
                // All headers except Authorization
            },
            "params": {
                "limit": ["100"],
                "offset": ["0"]
                // Query parameters as key -> list of values
            }
        }
    }
    ```

    Expected Auth Endpoint Response Format:
    ```json
    {
        "access_attributes": {    // Structured attribute format
            "roles": ["admin", "user"],
            "teams": ["ml-team", "nlp-team"],
            "projects": ["llama-3", "project-x"],
            "namespaces": ["research"]
        },
        "message": "Optional message about auth result"
    }
    ```

    Token Validation:
    Each provider implements its own token validation logic:
    - Kubernetes: Uses TokenReview API to validate service account tokens
    - Custom: Sends token to custom endpoint for validation

    Attribute-Based Access Control:
    The attributes returned by the auth provider are used to determine which
    resources the user can access. Resources can specify required attributes
    using the access_attributes field. For a user to access a resource:

    1. All attribute categories specified in the resource must be present in the user's attributes
    2. For each category, the user must have at least one matching value

    If the auth provider doesn't return any attributes, the user will only be able to
    access resources that don't have access_attributes defined.
    """

    def __init__(self, app, auth_config: AuthenticationConfig, impls):
        self.app = app
        self.impls = impls
        self.auth_provider = create_auth_provider(auth_config)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Check if authentication is required
            path = scope.get("path", "")

            # Known public endpoints that don't require authentication
            # These are typically operational endpoints like health checks
            public_paths = (
                "/v1/health",
                "/v1/inspect/version",
            )

            # Check if this is a public endpoint
            if (
                path in public_paths
                or path.startswith("/docs")
                or path.startswith("/redoc")
                or path.startswith("/openapi.json")
            ):
                logger.debug(f"Allowing unauthenticated access to public endpoint: {path}")
                return await self.app(scope, receive, send)

            # Handle authentication
            headers = dict(scope.get("headers", []))
            auth_header = headers.get(b"authorization", b"").decode()

            if not auth_header:
                error_msg = self.auth_provider.get_auth_error_message(scope)
                return await self._send_auth_error(send, error_msg)

            if not auth_header.startswith("Bearer "):
                return await self._send_auth_error(send, "Invalid Authorization header format")

            token = auth_header.split("Bearer ", 1)[1]

            # Validate token and get access attributes
            try:
                validation_result = await self.auth_provider.validate_token(token, scope)
            except httpx.TimeoutException:
                logger.exception("Authentication request timed out")
                return await self._send_auth_error(send, "Authentication service timeout")
            except ValueError as e:
                logger.exception("Error during authentication")
                return await self._send_auth_error(send, str(e))
            except Exception:
                logger.exception("Error during authentication")
                return await self._send_auth_error(send, "Authentication service error")

            # Store the client ID in the request scope so that downstream middleware (like QuotaMiddleware)
            # can identify the requester and enforce per-client rate limits.
            scope["authenticated_client_id"] = token

            # Store attributes in request scope
            scope["principal"] = validation_result.principal
            if validation_result.attributes:
                scope["user_attributes"] = validation_result.attributes
            logger.debug(
                f"Authentication successful: {validation_result.principal} with {len(validation_result.attributes)} attributes"
            )

            # Note: Scope-based API access control can be implemented at the route level
            # using FastAPI dependencies if needed

        return await self.app(scope, receive, send)

    async def _send_auth_error(self, send, message, status=401):
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [[b"content-type", b"application/json"]],
            }
        )
        error_key = "message" if status == 401 else "detail"
        error_msg = json.dumps({"error": {error_key: message}}).encode()
        await send({"type": "http.response.body", "body": error_msg})

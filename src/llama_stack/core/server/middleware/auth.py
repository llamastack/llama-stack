# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json

import httpx
from aiohttp import hdrs

from llama_stack.core.access_control.conditions import parse_conditions
from llama_stack.core.access_control.datatypes import RouteAccessRule
from llama_stack.core.datatypes import AuthenticationConfig, User
from llama_stack.core.request_headers import user_from_scope
from llama_stack.core.server.auth_providers import create_auth_provider
from llama_stack.core.server.routes import find_matching_route, initialize_route_impls
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
    Endpoints can opt out of authentication by:
    - For legacy @webmethod routes: setting require_authentication=False in the decorator
    - For FastAPI router routes: setting openapi_extra={PUBLIC_ROUTE_KEY: True}
    This is typically used for operational endpoints like /health and /version to support
    monitoring, load balancers, and observability tools.

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
            # Find the route and check if authentication is required
            path = scope.get("path", "")
            method = scope.get("method", hdrs.METH_GET)

            if not hasattr(self, "route_impls"):
                self.route_impls = initialize_route_impls(self.impls)

            webmethod = None
            try:
                _, _, _, webmethod = find_matching_route(method, path, self.route_impls)
            except ValueError:
                # If no matching endpoint is found, pass here to run auth anyways
                pass

            # If webmethod explicitly sets require_authentication=False, allow without auth
            if webmethod and webmethod.require_authentication is False:
                logger.debug(f"Allowing unauthenticated access to endpoint: {path}")
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


class RouteAuthorizationMiddleware:
    """Middleware that enforces route-level access control.

    This middleware runs after authentication and checks if the authenticated user
    has permission to access the requested API route based on route_policy rules.

    """

    def __init__(self, app, route_policy: list[RouteAccessRule]):
        self.app = app
        self.route_policy = route_policy

    async def __call__(self, scope, receive, send):
        # Only process HTTP requests
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        # If no route policy configured, allow all routes (backward compatible)
        if not self.route_policy:
            return await self.app(scope, receive, send)

        route = scope.get("path", "")
        # Normalize route: remove trailing slash (except for root "/")
        if route != "/" and route.endswith("/"):
            route = route.rstrip("/")

        # Get authenticated user from scope (set by AuthenticationMiddleware if present)
        user = user_from_scope(scope)

        # Check if user has permission to access this route
        if not self._is_route_allowed(route, user):
            return await self._send_error(
                send, f"Access denied: insufficient permissions for route {route}", status=403
            )

        return await self.app(scope, receive, send)

    def _is_route_allowed(self, route: str, user: User | None) -> bool:
        """Check if the user is allowed to access the given route.

        Rules are evaluated in order. First matching rule determines access.
        If no rule matches, access is denied.

        Args:
            route: The route being accessed
            user: The authenticated user, or None if no authentication is configured
        """
        user_str = user.principal if user else "anonymous"

        for index, rule in enumerate(self.route_policy):
            if self._rule_matches(rule, route, user):
                # Check if this is a permit or forbid rule
                if rule.permit:
                    decision = "APPROVED"
                    reason = rule.description or ""
                    logger.debug(
                        f"ROUTE_AUTHZ,decision={decision},user={user_str},"
                        f"route={route},rule_index={index},reason={reason!r}"
                    )
                    return True
                else:  # forbid
                    decision = "DENIED"
                    reason = rule.description or ""
                    logger.debug(
                        f"ROUTE_AUTHZ,decision={decision},user={user_str},"
                        f"route={route},rule_index={index},reason={reason!r}"
                    )
                    return False

        # No matching rule found - deny by default
        decision = "DENIED"
        reason = "no matching rule"
        logger.debug(f"ROUTE_AUTHZ,decision={decision},user={user_str},route={route},rule_index=-1,reason={reason!r}")
        return False

    def _rule_matches(self, rule: RouteAccessRule, route: str, user: User | None) -> bool:
        """Check if a rule matches the given route and user.

        Args:
            rule: The rule to evaluate
            route: The route being accessed
            user: The authenticated user, or None if no authentication is configured
        """
        # Get the scope (permit or forbid)
        scope = rule.permit if rule.permit else rule.forbid
        if not scope:
            return False

        # Check if route matches
        if not self._route_matches(route, scope.paths):
            return False

        # Evaluate conditions
        return self._evaluate_conditions(rule, user)

    def _route_matches(self, request_route: str, rule_patterns: str | list[str]) -> bool:
        """Check if request route matches any of the rule patterns.

        Supports:
        - Exact match: "/v1/chat/completions"
        - Prefix wildcard: "/v1/files*" matches "/v1/files", "/v1/files/upload", "/v1/files/list", etc.
        - Full wildcard: "*" matches all routes
        """
        patterns = [rule_patterns] if isinstance(rule_patterns, str) else rule_patterns

        for pattern in patterns:
            if pattern == "*":
                # Full wildcard matches everything
                return True
            elif pattern.endswith("*"):
                # Prefix wildcard: check if request route starts with the prefix
                prefix = pattern[:-1]  # Remove "*"
                if request_route.startswith(prefix):
                    return True
            elif pattern == request_route:
                # Exact match
                return True

        return False

    def _evaluate_conditions(self, rule: RouteAccessRule, user: User | None) -> bool:
        """Evaluate when/unless conditions for the rule.

        Reuses the existing condition parsing from access_control.conditions.

        Args:
            rule: The rule whose conditions to evaluate
            user: The authenticated user, or None if no authentication is configured

        Returns:
            True if conditions are met (or no conditions exist), False otherwise
        """
        # If rule has conditions but no user is available, conditions cannot be met
        if (rule.when or rule.unless) and not user:
            return False

        if rule.when:
            # At this point, if rule.when exists and we got past the check above,
            # user is guaranteed to be non-None
            assert user is not None
            conditions_list = rule.when if isinstance(rule.when, list) else [rule.when]
            conditions = parse_conditions(conditions_list)
            # For 'when', all conditions must match (AND logic)
            # Note: Since we're checking route access, we don't have a resource,
            # so we create a context object to satisfy the interface
            route_context = _RouteContext()
            for condition in conditions:
                if not condition.matches(route_context, user):
                    return False
            return True

        if rule.unless:
            # At this point, if rule.unless exists and we got past the check above,
            # user is guaranteed to be non-None
            assert user is not None
            conditions_list = rule.unless if isinstance(rule.unless, list) else [rule.unless]
            conditions = parse_conditions(conditions_list)
            # For 'unless', no conditions should match (NOT logic)
            route_context = _RouteContext()
            for condition in conditions:
                if condition.matches(route_context, user):
                    return False
            return True

        # No conditions specified - rule applies regardless of user
        return True

    async def _send_error(self, send, message: str, status: int = 403):
        """Send an error response."""
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


class _RouteContext:
    """Placeholder resource for route-level condition evaluation.

    Route rules don't operate on actual resources, so we use this context object
    to satisfy the condition.matches() interface. Route conditions typically check
    user attributes (e.g., "user with admin in roles") and don't require resource properties.
    """

    def __init__(self):
        self.type = "route"
        self.identifier = "route"
        self.owner = None

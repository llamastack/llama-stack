# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import Mock

import pytest
from fastapi import FastAPI

from llama_stack.core.access_control.datatypes import EndpointAccessRule, EndpointScope
from llama_stack.core.datatypes import (
    AuthenticationConfig,
    AuthProviderType,
    CustomAuthConfig,
    User,
)
from llama_stack.core.server.auth import (
    AuthenticationMiddleware,
    EndpointAuthorizationMiddleware,
)


@pytest.fixture
def admin_user():
    return User(
        principal="admin@example.com",
        attributes={
            "roles": ["admin"],
            "teams": ["platform"],
        },
    )


@pytest.fixture
def developer_user():
    return User(
        principal="dev@example.com",
        attributes={
            "roles": ["developer"],
            "teams": ["ml-team"],
        },
    )


@pytest.fixture
def regular_user():
    return User(
        principal="user@example.com",
        attributes={
            "roles": ["user"],
            "teams": ["ml-team"],
        },
    )


def create_mock_auth_provider(user: User):
    """Create a mock auth provider that returns the specified user"""
    mock_provider = Mock()
    mock_provider.validate_token = Mock(return_value=user)
    return mock_provider


def create_app_with_endpoint_policy(endpoint_policy: list[EndpointAccessRule], user: User):
    """Create a FastAPI app with endpoint authorization middleware"""
    app = FastAPI()

    # Create auth config
    auth_config = AuthenticationConfig(
        provider_config=CustomAuthConfig(
            type=AuthProviderType.CUSTOM,
            endpoint="http://mock-auth/validate",
        ),
        endpoint_policy=endpoint_policy,
        access_policy=[],
    )

    # Add authentication middleware
    auth_middleware = AuthenticationMiddleware(app, auth_config, {})
    # Mock the auth provider to return our test user
    auth_middleware.auth_provider = create_mock_auth_provider(user)

    # Create the middleware stack
    app.add_middleware(EndpointAuthorizationMiddleware, endpoint_policy=endpoint_policy)

    # Replace the app in the auth middleware
    async def app_with_auth(scope, receive, send):
        return await auth_middleware(scope, receive, send)

    # Add test endpoints
    @app.get("/v1/chat/completions")
    def chat_completions():
        return {"message": "chat completions"}

    @app.get("/v1/models/list")
    def models_list():
        return {"message": "models list"}

    @app.post("/v1/files/upload")
    def files_upload():
        return {"message": "file uploaded"}

    @app.delete("/v1/admin/reset")
    def admin_reset():
        return {"message": "admin reset"}

    return app, app_with_auth


async def test_no_endpoint_policy_allows_all(regular_user):
    """Test backward compatibility: empty endpoint_policy allows all endpoints"""
    app = FastAPI()

    # No endpoint policy
    endpoint_policy = []

    middleware = EndpointAuthorizationMiddleware(app, endpoint_policy)

    # Mock scope with any path
    scope = {
        "type": "http",
        "path": "/v1/chat/completions",
        "method": "GET",
        "principal": regular_user.principal,
        "user_attributes": regular_user.attributes,
    }

    # Track if next middleware was called
    called = False

    async def mock_app(scope, receive, send):
        nonlocal called
        called = True

    async def receive():
        return {}

    async def send(msg):
        pass

    middleware.app = mock_app
    await middleware(scope, receive, send)

    # Should pass through without blocking
    assert called


async def test_exact_path_match(developer_user):
    """Test exact path matching"""
    endpoint_policy = [
        EndpointAccessRule(
            permit=EndpointScope(paths="/v1/chat/completions"),
            when="user with developer in roles",
        )
    ]

    middleware = EndpointAuthorizationMiddleware(None, endpoint_policy)

    # Should match
    assert middleware._path_matches("/v1/chat/completions", "/v1/chat/completions")

    # Should not match
    assert not middleware._path_matches("/v1/chat/completions/stream", "/v1/chat/completions")
    assert not middleware._path_matches("/v1/models/list", "/v1/chat/completions")


async def test_wildcard_prefix_match():
    """Test wildcard prefix matching"""
    endpoint_policy = []
    middleware = EndpointAuthorizationMiddleware(None, endpoint_policy)

    # Test prefix wildcard
    assert middleware._path_matches("/v1/files/upload", "/v1/files*")
    assert middleware._path_matches("/v1/files/delete", "/v1/files*")
    assert middleware._path_matches("/v1/files/list/all", "/v1/files*")
    # Should also match the exact prefix
    assert middleware._path_matches("/v1/files", "/v1/files*")

    # Should not match different prefix
    assert not middleware._path_matches("/v1/models/list", "/v1/files*")


async def test_full_wildcard_match():
    """Test full wildcard matching"""
    endpoint_policy = []
    middleware = EndpointAuthorizationMiddleware(None, endpoint_policy)

    # Full wildcard should match everything
    assert middleware._path_matches("/v1/chat/completions", "*")
    assert middleware._path_matches("/v1/files/upload", "*")
    assert middleware._path_matches("/admin/reset", "*")
    assert middleware._path_matches("/anything/goes", "*")


async def test_multiple_paths_in_rule(regular_user):
    """Test rule with multiple paths"""
    endpoint_policy = [
        EndpointAccessRule(
            permit=EndpointScope(paths=["/v1/files*", "/v1/models*"]),
            when="user with user in roles",
        )
    ]

    middleware = EndpointAuthorizationMiddleware(None, endpoint_policy)

    # Test that the policy allows these paths for regular_user
    assert middleware._is_endpoint_allowed("/v1/files/upload", regular_user)
    assert middleware._is_endpoint_allowed("/v1/models/list", regular_user)

    # Should not match other paths
    assert not middleware._is_endpoint_allowed("/v1/chat/completions", regular_user)


async def test_condition_evaluation_with_roles(developer_user, regular_user):
    """Test condition evaluation with role attributes"""
    endpoint_policy = [
        EndpointAccessRule(
            permit=EndpointScope(paths="/v1/chat/completions"),
            when="user with developer in roles",
        )
    ]

    middleware = EndpointAuthorizationMiddleware(None, endpoint_policy)

    # Developer should pass
    assert middleware._is_endpoint_allowed("/v1/chat/completions", developer_user)

    # Regular user should not pass
    assert not middleware._is_endpoint_allowed("/v1/chat/completions", regular_user)


async def test_admin_full_wildcard_access(admin_user, developer_user):
    """Test admin with full wildcard access"""
    endpoint_policy = [
        EndpointAccessRule(
            permit=EndpointScope(paths="/v1/chat/completions"),
            when="user with developer in roles",
        ),
        EndpointAccessRule(
            permit=EndpointScope(paths="*"),
            when="user with admin in roles",
        ),
    ]

    middleware = EndpointAuthorizationMiddleware(None, endpoint_policy)

    # Admin should access everything
    assert middleware._is_endpoint_allowed("/v1/chat/completions", admin_user)
    assert middleware._is_endpoint_allowed("/v1/files/upload", admin_user)
    assert middleware._is_endpoint_allowed("/v1/admin/reset", admin_user)

    # Developer should only access chat completions
    assert middleware._is_endpoint_allowed("/v1/chat/completions", developer_user)
    assert not middleware._is_endpoint_allowed("/v1/files/upload", developer_user)


async def test_forbid_rule(admin_user, developer_user):
    """Test forbid rules with unless conditions"""
    endpoint_policy = [
        EndpointAccessRule(
            forbid=EndpointScope(paths="/v1/admin*"),
            unless="user with admin in roles",
        ),
        EndpointAccessRule(
            permit=EndpointScope(paths="*"),
            when="user with admin in roles",
            description="Admins can access everything",
        ),
        EndpointAccessRule(
            permit=EndpointScope(paths="*"),
            when="user with developer in roles",
            description="Developers can access everything except what's forbidden",
        ),
    ]

    middleware = EndpointAuthorizationMiddleware(None, endpoint_policy)

    # Developer is forbidden from admin endpoints by the first forbid rule
    assert not middleware._is_endpoint_allowed("/v1/admin/reset", developer_user)

    # Admin bypasses the forbid rule (due to 'unless' condition) and matches the permit rule
    assert middleware._is_endpoint_allowed("/v1/admin/reset", admin_user)

    # Developer can access non-admin endpoints
    assert middleware._is_endpoint_allowed("/v1/chat/completions", developer_user)


async def test_no_matching_rule_denies_access(regular_user):
    """Test that no matching rule results in denied access"""
    endpoint_policy = [
        EndpointAccessRule(
            permit=EndpointScope(paths="/v1/chat/completions"),
            when="user with developer in roles",
        )
    ]

    middleware = EndpointAuthorizationMiddleware(None, endpoint_policy)

    # Regular user should be denied (doesn't have developer role)
    assert not middleware._is_endpoint_allowed("/v1/chat/completions", regular_user)

    # Any user should be denied for non-matching path
    assert not middleware._is_endpoint_allowed("/v1/models/list", regular_user)


async def test_multiple_conditions(admin_user):
    """Test multiple conditions in a rule"""
    endpoint_policy = [
        EndpointAccessRule(
            permit=EndpointScope(paths="*"),
            when=["user with admin in roles", "user with platform in teams"],
        )
    ]

    middleware = EndpointAuthorizationMiddleware(None, endpoint_policy)

    # Admin with platform team should pass
    assert middleware._is_endpoint_allowed("/v1/anything", admin_user)


async def test_rule_order_matters(developer_user):
    """Test that rules are evaluated in order"""
    endpoint_policy = [
        # First rule: developers can access chat
        EndpointAccessRule(
            permit=EndpointScope(paths="/v1/chat/completions"),
            when="user with developer in roles",
        ),
        # Second rule: deny all developers from everything (should not apply to chat)
        EndpointAccessRule(
            forbid=EndpointScope(paths="*"),
            when="user with developer in roles",
        ),
    ]

    middleware = EndpointAuthorizationMiddleware(None, endpoint_policy)

    # First matching rule should win
    assert middleware._is_endpoint_allowed("/v1/chat/completions", developer_user)

    # Second rule should match for other paths
    assert not middleware._is_endpoint_allowed("/v1/models/list", developer_user)


async def test_websocket_passthrough():
    """Test that websocket requests pass through without blocking"""
    endpoint_policy = [
        EndpointAccessRule(
            permit=EndpointScope(paths="/v1/chat/completions"),
            when="user with developer in roles",
        )
    ]

    app = FastAPI()
    middleware = EndpointAuthorizationMiddleware(app, endpoint_policy)

    # Mock websocket scope
    scope = {
        "type": "websocket",
        "path": "/ws",
    }

    # Track if next middleware was called
    called = False

    async def mock_app(scope, receive, send):
        nonlocal called
        called = True

    async def receive():
        return {}

    async def send(msg):
        pass

    middleware.app = mock_app
    await middleware(scope, receive, send)

    # Websocket requests should pass through
    assert called


async def test_endpoint_blocking_without_auth():
    """Test that endpoint policy can block endpoints without authentication configured"""
    endpoint_policy = [
        EndpointAccessRule(
            permit=EndpointScope(paths="/v1/health"),
            description="Allow health check endpoint",
        ),
        EndpointAccessRule(
            permit=EndpointScope(paths="/v1/models*"),
            description="Allow model endpoints",
        ),
        # All other endpoints denied by default (no matching rule)
    ]

    app = FastAPI()
    middleware = EndpointAuthorizationMiddleware(app, endpoint_policy)

    # No user (no authentication)
    user = None

    # Should allow health check
    assert middleware._is_endpoint_allowed("/v1/health", user)

    # Should allow model endpoints
    assert middleware._is_endpoint_allowed("/v1/models/list", user)

    # Should deny other endpoints (no matching rule)
    assert not middleware._is_endpoint_allowed("/v1/chat/completions", user)
    assert not middleware._is_endpoint_allowed("/v1/admin/reset", user)


async def test_forbid_rule_without_auth():
    """Test forbid rules work without authentication"""
    endpoint_policy = [
        EndpointAccessRule(
            forbid=EndpointScope(paths="/v1/admin*"),
            description="Block admin endpoints",
        ),
        EndpointAccessRule(
            permit=EndpointScope(paths="*"),
            description="Allow all other endpoints",
        ),
    ]

    middleware = EndpointAuthorizationMiddleware(None, endpoint_policy)

    # No user (no authentication)
    user = None

    # Should forbid admin endpoints
    assert not middleware._is_endpoint_allowed("/v1/admin/reset", user)
    assert not middleware._is_endpoint_allowed("/v1/admin/users", user)

    # Should allow other endpoints
    assert middleware._is_endpoint_allowed("/v1/chat/completions", user)
    assert middleware._is_endpoint_allowed("/v1/models/list", user)


async def test_rule_with_condition_requires_user():
    """Test that rules with user conditions require authentication"""
    endpoint_policy = [
        EndpointAccessRule(
            permit=EndpointScope(paths="*"),
            when="user with admin in roles",
        )
    ]

    middleware = EndpointAuthorizationMiddleware(None, endpoint_policy)

    # No user (no authentication)
    user = None

    # Should be denied because rule has condition but no user available
    assert not middleware._is_endpoint_allowed("/v1/chat/completions", user)


async def test_mixed_rules_with_and_without_conditions(admin_user, regular_user):
    """Test mixing rules with and without user conditions"""
    endpoint_policy = [
        # Public endpoints (no condition)
        EndpointAccessRule(
            permit=EndpointScope(paths=["/v1/health", "/v1/version"]),
            description="Public endpoints",
        ),
        # Admin-only endpoints (requires user)
        EndpointAccessRule(
            permit=EndpointScope(paths="/v1/admin*"),
            when="user with admin in roles",
            description="Admin endpoints require admin role",
        ),
        # Default: deny everything else
    ]

    middleware = EndpointAuthorizationMiddleware(None, endpoint_policy)

    # No user can access public endpoints
    assert middleware._is_endpoint_allowed("/v1/health", None)
    assert middleware._is_endpoint_allowed("/v1/version", None)

    # No user cannot access admin endpoints (condition requires user)
    assert not middleware._is_endpoint_allowed("/v1/admin/reset", None)

    # Admin can access admin endpoints
    assert middleware._is_endpoint_allowed("/v1/admin/reset", admin_user)

    # Regular user cannot access admin endpoints (lacks admin role)
    assert not middleware._is_endpoint_allowed("/v1/admin/reset", regular_user)

    # No one can access other endpoints (no matching rule)
    assert not middleware._is_endpoint_allowed("/v1/chat/completions", None)
    assert not middleware._is_endpoint_allowed("/v1/chat/completions", admin_user)
    assert not middleware._is_endpoint_allowed("/v1/chat/completions", regular_user)

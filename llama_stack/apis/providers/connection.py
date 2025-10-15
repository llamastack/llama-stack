# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from llama_stack.core.datatypes import User
from llama_stack.providers.datatypes import HealthStatus
from llama_stack.schema_utils import json_schema_type


@json_schema_type
class ProviderConnectionStatus(StrEnum):
    """Status of a dynamic provider connection.

    :cvar pending: Configuration stored, not yet initialized
    :cvar initializing: In the process of connecting
    :cvar connected: Successfully connected and healthy
    :cvar failed: Connection attempt failed
    :cvar disconnected: Previously connected, now disconnected
    :cvar testing: Health check in progress
    """

    pending = "pending"
    initializing = "initializing"
    connected = "connected"
    failed = "failed"
    disconnected = "disconnected"
    testing = "testing"


@json_schema_type
class ProviderHealth(BaseModel):
    """Structured wrapper around provider health status.

    This wraps the existing dict-based HealthResponse for API responses
    while maintaining backward compatibility with existing provider implementations.

    :param status: Health status (OK, ERROR, NOT_IMPLEMENTED)
    :param message: Optional error or status message
    :param metrics: Provider-specific health metrics
    :param last_checked: Timestamp of last health check
    """

    status: HealthStatus
    message: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    last_checked: datetime

    @classmethod
    def from_health_response(cls, response: dict[str, Any]) -> "ProviderHealth":
        """Convert dict-based HealthResponse to ProviderHealth.

        This allows us to maintain the existing dict[str, Any] return type
        for provider.health() methods while providing a structured model
        for API responses.

        :param response: Dict with 'status' and optional 'message', 'metrics'
        :returns: ProviderHealth instance
        """
        return cls(
            status=HealthStatus(response.get("status", HealthStatus.NOT_IMPLEMENTED)),
            message=response.get("message"),
            metrics=response.get("metrics", {}),
            last_checked=datetime.now(UTC),
        )


@json_schema_type
class ProviderConnectionInfo(BaseModel):
    """Information about a dynamically managed provider connection.

    This model represents a provider that has been registered at runtime
    via the /providers API, as opposed to static providers configured in run.yaml.

    Dynamic providers support full lifecycle management including registration,
    configuration updates, health monitoring, and removal.

    :param provider_id: Unique identifier for this provider instance
    :param api: API namespace (e.g., "inference", "vector_io", "safety")
    :param provider_type: Provider type identifier (e.g., "remote::openai", "inline::faiss")
    :param config: Provider-specific configuration (API keys, endpoints, etc.)
    :param status: Current connection status
    :param health: Most recent health check result
    :param created_at: Timestamp when provider was registered
    :param updated_at: Timestamp of last update
    :param last_health_check: Timestamp of last health check
    :param error_message: Error message if status is failed
    :param metadata: User-defined metadata (deprecated, use attributes)
    :param owner: User who created this provider connection
    :param attributes: Key-value attributes for ABAC access control
    """

    provider_id: str
    api: str
    provider_type: str
    config: dict[str, Any]
    status: ProviderConnectionStatus
    health: ProviderHealth | None = None
    created_at: datetime
    updated_at: datetime
    last_health_check: datetime | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Deprecated: use attributes for access control",
    )

    # ABAC fields (same as ResourceWithOwner)
    owner: User | None = None
    attributes: dict[str, list[str]] | None = None

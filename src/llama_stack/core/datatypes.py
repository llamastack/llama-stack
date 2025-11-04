# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Self
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator

from llama_stack.apis.benchmarks import Benchmark, BenchmarkInput
from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Dataset, DatasetInput
from llama_stack.apis.eval import Eval
from llama_stack.apis.inference import Inference
from llama_stack.apis.models import Model, ModelInput
from llama_stack.apis.resource import Resource
from llama_stack.apis.safety import Safety
from llama_stack.apis.scoring import Scoring
from llama_stack.apis.scoring_functions import ScoringFn, ScoringFnInput
from llama_stack.apis.shields import Shield, ShieldInput
from llama_stack.apis.tools import ToolGroup, ToolGroupInput, ToolRuntime
from llama_stack.apis.vector_io import VectorIO
from llama_stack.apis.vector_stores import VectorStore, VectorStoreInput
from llama_stack.core.access_control.datatypes import AccessRule
from llama_stack.core.storage.datatypes import (
    KVStoreReference,
    StorageBackendType,
    StorageConfig,
)
from llama_stack.log import LoggingConfig
from llama_stack.providers.datatypes import Api, ProviderSpec

LLAMA_STACK_BUILD_CONFIG_VERSION = 2
LLAMA_STACK_RUN_CONFIG_VERSION = 2


RoutingKey = str | list[str]


class RegistryEntrySource(StrEnum):
    via_register_api = "via_register_api"
    listed_from_provider = "listed_from_provider"


class User(BaseModel):
    principal: str
    # further attributes that may be used for access control decisions
    attributes: dict[str, list[str]] | None = None

    def __init__(self, principal: str, attributes: dict[str, list[str]] | None):
        super().__init__(principal=principal, attributes=attributes)


class ResourceWithOwner(Resource):
    """Extension of Resource that adds an optional owner, i.e. the user that created the
    resource. This can be used to constrain access to the resource."""

    owner: User | None = None
    source: RegistryEntrySource = RegistryEntrySource.via_register_api


# Use the extended Resource for all routable objects
class ModelWithOwner(Model, ResourceWithOwner):
    pass


class ShieldWithOwner(Shield, ResourceWithOwner):
    pass


class VectorStoreWithOwner(VectorStore, ResourceWithOwner):
    pass


class DatasetWithOwner(Dataset, ResourceWithOwner):
    pass


class ScoringFnWithOwner(ScoringFn, ResourceWithOwner):
    pass


class BenchmarkWithOwner(Benchmark, ResourceWithOwner):
    pass


class ToolGroupWithOwner(ToolGroup, ResourceWithOwner):
    pass


RoutableObject = Model | Shield | VectorStore | Dataset | ScoringFn | Benchmark | ToolGroup

RoutableObjectWithProvider = Annotated[
    ModelWithOwner
    | ShieldWithOwner
    | VectorStoreWithOwner
    | DatasetWithOwner
    | ScoringFnWithOwner
    | BenchmarkWithOwner
    | ToolGroupWithOwner,
    Field(discriminator="type"),
]

RoutedProtocol = Inference | Safety | VectorIO | DatasetIO | Scoring | Eval | ToolRuntime


# Example: /inference, /safety
class AutoRoutedProviderSpec(ProviderSpec):
    provider_type: str = "router"
    config_class: str = ""

    container_image: str | None = None
    routing_table_api: Api
    module: str
    provider_data_validator: str | None = Field(
        default=None,
    )


# Example: /models, /shields
class RoutingTableProviderSpec(ProviderSpec):
    provider_type: str = "routing_table"
    config_class: str = ""
    container_image: str | None = None

    router_api: Api
    module: str
    pip_packages: list[str] = Field(default_factory=list)


class Provider(BaseModel):
    # provider_id of None means that the provider is not enabled - this happens
    # when the provider is enabled via a conditional environment variable
    provider_id: str | None
    provider_type: str
    config: dict[str, Any] = {}
    module: str | None = Field(
        default=None,
        description="""
 Fully-qualified name of the external provider module to import. The module is expected to have:

  - `get_adapter_impl(config, deps)`: returns the adapter implementation

  Example: `module: ramalama_stack`
 """,
    )


class BuildProvider(BaseModel):
    provider_type: str
    module: str | None = Field(
        default=None,
        description="""
 Fully-qualified name of the external provider module to import. The module is expected to have:

  - `get_adapter_impl(config, deps)`: returns the adapter implementation

  Example: `module: ramalama_stack`
 """,
    )


class DistributionSpec(BaseModel):
    description: str | None = Field(
        default="",
        description="Description of the distribution",
    )
    container_image: str | None = None
    providers: dict[str, list[BuildProvider]] = Field(
        default_factory=dict,
        description="""
        Provider Types for each of the APIs provided by this distribution. If you
        select multiple providers, you should provide an appropriate 'routing_map'
        in the runtime configuration to help route to the correct provider.
        """,
    )


class TelemetryConfig(BaseModel):
    """
    Configuration for telemetry.

    Llama Stack uses OpenTelemetry for telemetry. Please refer to https://opentelemetry.io/docs/languages/sdk-configuration/
    for env variables to configure the OpenTelemetry SDK.

    Example:
    ```bash
    OTEL_SERVICE_NAME=llama-stack OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 uv run llama stack run starter
    ```
    """

    enabled: bool = Field(default=False, description="enable or disable telemetry")


class OAuth2JWKSConfig(BaseModel):
    # The JWKS URI for collecting public keys
    uri: str
    token: str | None = Field(default=None, description="token to authorise access to jwks")
    key_recheck_period: int = Field(default=3600, description="The period to recheck the JWKS URI for key updates")


class OAuth2IntrospectionConfig(BaseModel):
    url: str
    client_id: str
    client_secret: str
    send_secret_in_body: bool = False


class AuthProviderType(StrEnum):
    """Supported authentication provider types."""

    OAUTH2_TOKEN = "oauth2_token"
    GITHUB_TOKEN = "github_token"
    CUSTOM = "custom"
    KUBERNETES = "kubernetes"


class OAuth2TokenAuthConfig(BaseModel):
    """Configuration for OAuth2 token authentication."""

    type: Literal[AuthProviderType.OAUTH2_TOKEN] = AuthProviderType.OAUTH2_TOKEN
    audience: str = Field(default="llama-stack")
    verify_tls: bool = Field(default=True)
    tls_cafile: Path | None = Field(default=None)
    issuer: str | None = Field(default=None, description="The OIDC issuer URL.")
    claims_mapping: dict[str, str] = Field(
        default_factory=lambda: {
            "sub": "roles",
            "username": "roles",
            "groups": "teams",
            "team": "teams",
            "project": "projects",
            "tenant": "namespaces",
            "namespace": "namespaces",
        },
    )
    jwks: OAuth2JWKSConfig | None = Field(default=None, description="JWKS configuration")
    introspection: OAuth2IntrospectionConfig | None = Field(
        default=None, description="OAuth2 introspection configuration"
    )

    @classmethod
    @field_validator("claims_mapping")
    def validate_claims_mapping(cls, v):
        for key, value in v.items():
            if not value:
                raise ValueError(f"claims_mapping value cannot be empty: {key}")
        return v

    @model_validator(mode="after")
    def validate_mode(self) -> Self:
        if not self.jwks and not self.introspection:
            raise ValueError("One of jwks or introspection must be configured")
        if self.jwks and self.introspection:
            raise ValueError("At present only one of jwks or introspection should be configured")
        return self


class CustomAuthConfig(BaseModel):
    """Configuration for custom authentication."""

    type: Literal[AuthProviderType.CUSTOM] = AuthProviderType.CUSTOM
    endpoint: str = Field(
        ...,
        description="Custom authentication endpoint URL",
    )


class GitHubTokenAuthConfig(BaseModel):
    """Configuration for GitHub token authentication."""

    type: Literal[AuthProviderType.GITHUB_TOKEN] = AuthProviderType.GITHUB_TOKEN
    github_api_base_url: str = Field(
        default="https://api.github.com",
        description="Base URL for GitHub API (use https://api.github.com for public GitHub)",
    )
    claims_mapping: dict[str, str] = Field(
        default_factory=lambda: {
            "login": "roles",
            "organizations": "teams",
        },
        description="Mapping from GitHub user fields to access attributes",
    )


class KubernetesAuthProviderConfig(BaseModel):
    """Configuration for Kubernetes authentication provider."""

    type: Literal[AuthProviderType.KUBERNETES] = AuthProviderType.KUBERNETES
    api_server_url: str = Field(
        default="https://kubernetes.default.svc",
        description="Kubernetes API server URL (e.g., https://api.cluster.domain:6443)",
    )
    verify_tls: bool = Field(default=True, description="Whether to verify TLS certificates")
    tls_cafile: Path | None = Field(default=None, description="Path to CA certificate file for TLS verification")
    claims_mapping: dict[str, str] = Field(
        default_factory=lambda: {
            "username": "roles",
            "groups": "roles",
        },
        description="Mapping of Kubernetes user claims to access attributes",
    )

    @field_validator("api_server_url")
    @classmethod
    def validate_api_server_url(cls, v):
        parsed = urlparse(v)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"api_server_url must be a valid URL with scheme and host: {v}")
        if parsed.scheme not in ["http", "https"]:
            raise ValueError(f"api_server_url scheme must be http or https: {v}")
        return v

    @field_validator("claims_mapping")
    @classmethod
    def validate_claims_mapping(cls, v):
        for key, value in v.items():
            if not value:
                raise ValueError(f"claims_mapping value cannot be empty: {key}")
        return v


AuthProviderConfig = Annotated[
    OAuth2TokenAuthConfig | GitHubTokenAuthConfig | CustomAuthConfig | KubernetesAuthProviderConfig,
    Field(discriminator="type"),
]


class AuthenticationConfig(BaseModel):
    """Top-level authentication configuration."""

    provider_config: AuthProviderConfig = Field(
        ...,
        description="Authentication provider configuration",
    )
    access_policy: list[AccessRule] = Field(
        default=[],
        description="Rules for determining access to resources",
    )


class AuthenticationRequiredError(Exception):
    pass


class QualifiedModel(BaseModel):
    """A qualified model identifier, consisting of a provider ID and a model ID."""

    provider_id: str
    model_id: str


class VectorStoresConfig(BaseModel):
    """Configuration for vector stores in the stack."""

    default_provider_id: str | None = Field(
        default=None,
        description="ID of the vector_io provider to use as default when multiple providers are available and none is specified.",
    )
    default_embedding_model: QualifiedModel | None = Field(
        default=None,
        description="Default embedding model configuration for vector stores.",
    )


class SafetyConfig(BaseModel):
    """Configuration for default moderations model."""

    default_shield_id: str | None = Field(
        default=None,
        description="ID of the shield to use for when `model` is not specified in the `moderations` API request.",
    )


class QuotaPeriod(StrEnum):
    DAY = "day"


class QuotaConfig(BaseModel):
    kvstore: KVStoreReference = Field(description="Config for KV store backend (SQLite only for now)")
    anonymous_max_requests: int = Field(default=100, description="Max requests for unauthenticated clients per period")
    authenticated_max_requests: int = Field(
        default=1000, description="Max requests for authenticated clients per period"
    )
    period: QuotaPeriod = Field(default=QuotaPeriod.DAY, description="Quota period to set")


class CORSConfig(BaseModel):
    allow_origins: list[str] = Field(default_factory=list)
    allow_origin_regex: str | None = Field(default=None)
    allow_methods: list[str] = Field(default=["OPTIONS"])
    allow_headers: list[str] = Field(default_factory=list)
    allow_credentials: bool = Field(default=False)
    expose_headers: list[str] = Field(default_factory=list)
    max_age: int = Field(default=600, ge=0)

    @model_validator(mode="after")
    def validate_credentials_config(self) -> Self:
        if self.allow_credentials and (self.allow_origins == ["*"] or "*" in self.allow_origins):
            raise ValueError("Cannot use wildcard origins with credentials enabled")
        return self


def process_cors_config(cors_config: bool | CORSConfig | None) -> CORSConfig | None:
    if cors_config is False or cors_config is None:
        return None

    if cors_config is True:
        # dev mode: allow localhost on any port
        return CORSConfig(
            allow_origins=[],
            allow_origin_regex=r"https?://localhost:\d+",
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
        )

    if isinstance(cors_config, CORSConfig):
        return cors_config

    raise ValueError(f"Expected bool or CORSConfig, got {type(cors_config).__name__}")


class RegisteredResources(BaseModel):
    """Registry of resources available in the distribution."""

    models: list[ModelInput] = Field(default_factory=list)
    shields: list[ShieldInput] = Field(default_factory=list)
    vector_stores: list[VectorStoreInput] = Field(default_factory=list)
    datasets: list[DatasetInput] = Field(default_factory=list)
    scoring_fns: list[ScoringFnInput] = Field(default_factory=list)
    benchmarks: list[BenchmarkInput] = Field(default_factory=list)
    tool_groups: list[ToolGroupInput] = Field(default_factory=list)


class ServerConfig(BaseModel):
    port: int = Field(
        default=8321,
        description="Port to listen on",
        ge=1024,
        le=65535,
    )
    tls_certfile: str | None = Field(
        default=None,
        description="Path to TLS certificate file for HTTPS",
    )
    tls_keyfile: str | None = Field(
        default=None,
        description="Path to TLS key file for HTTPS",
    )
    tls_cafile: str | None = Field(
        default=None,
        description="Path to TLS CA file for HTTPS with mutual TLS authentication",
    )
    auth: AuthenticationConfig | None = Field(
        default=None,
        description="Authentication configuration for the server",
    )
    host: str | None = Field(
        default=None,
        description="The host the server should listen on",
    )
    quota: QuotaConfig | None = Field(
        default=None,
        description="Per client quota request configuration",
    )
    cors: bool | CORSConfig | None = Field(
        default=None,
        description="CORS configuration for cross-origin requests. Can be:\n"
        "- true: Enable localhost CORS for development\n"
        "- {allow_origins: [...], allow_methods: [...], ...}: Full configuration",
    )
    workers: int = Field(
        default=1,
        description="Number of workers to use for the server",
    )


class StackRunConfig(BaseModel):
    version: int = LLAMA_STACK_RUN_CONFIG_VERSION

    image_name: str = Field(
        ...,
        description="""
Reference to the distribution this package refers to. For unregistered (adhoc) packages,
this could be just a hash
""",
    )
    container_image: str | None = Field(
        default=None,
        description="Reference to the container image if this package refers to a container",
    )
    apis: list[str] = Field(
        default_factory=list,
        description="""
The list of APIs to serve. If not specified, all APIs specified in the provider_map will be served""",
    )

    providers: dict[str, list[Provider]] = Field(
        description="""
One or more providers to use for each API. The same provider_type (e.g., meta-reference)
can be instantiated multiple times (with different configs) if necessary.
""",
    )
    storage: StorageConfig = Field(
        description="Catalog of named storage backends and references available to the stack",
    )

    registered_resources: RegisteredResources = Field(
        default_factory=RegisteredResources,
        description="Registry of resources available in the distribution",
    )

    logging: LoggingConfig | None = Field(default=None, description="Configuration for Llama Stack Logging")

    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig, description="Configuration for telemetry")

    server: ServerConfig = Field(
        default_factory=ServerConfig,
        description="Configuration for the HTTP(S) server",
    )

    external_providers_dir: Path | None = Field(
        default=None,
        description="Path to directory containing external provider implementations. The providers code and dependencies must be installed on the system.",
    )

    external_apis_dir: Path | None = Field(
        default=None,
        description="Path to directory containing external API implementations. The APIs code and dependencies must be installed on the system.",
    )

    vector_stores: VectorStoresConfig | None = Field(
        default=None,
        description="Configuration for vector stores, including default embedding model",
    )

    safety: SafetyConfig | None = Field(
        default=None,
        description="Configuration for default moderations model",
    )

    @field_validator("external_providers_dir")
    @classmethod
    def validate_external_providers_dir(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            return Path(v)
        return v

    @model_validator(mode="after")
    def validate_server_stores(self) -> "StackRunConfig":
        backend_map = self.storage.backends
        stores = self.storage.stores
        kv_backends = {
            name
            for name, cfg in backend_map.items()
            if cfg.type
            in {
                StorageBackendType.KV_REDIS,
                StorageBackendType.KV_SQLITE,
                StorageBackendType.KV_POSTGRES,
                StorageBackendType.KV_MONGODB,
            }
        }
        sql_backends = {
            name
            for name, cfg in backend_map.items()
            if cfg.type in {StorageBackendType.SQL_SQLITE, StorageBackendType.SQL_POSTGRES}
        }

        def _ensure_backend(reference, expected_set, store_name: str) -> None:
            if reference is None:
                return
            backend_name = reference.backend
            if backend_name not in backend_map:
                raise ValueError(
                    f"{store_name} references unknown backend '{backend_name}'. "
                    f"Available backends: {sorted(backend_map)}"
                )
            if backend_name not in expected_set:
                raise ValueError(
                    f"{store_name} references backend '{backend_name}' of type "
                    f"'{backend_map[backend_name].type.value}', but a backend of type "
                    f"{'kv_*' if expected_set is kv_backends else 'sql_*'} is required."
                )

        _ensure_backend(stores.metadata, kv_backends, "storage.stores.metadata")
        _ensure_backend(stores.inference, sql_backends, "storage.stores.inference")
        _ensure_backend(stores.conversations, sql_backends, "storage.stores.conversations")
        _ensure_backend(stores.responses, sql_backends, "storage.stores.responses")
        _ensure_backend(stores.prompts, kv_backends, "storage.stores.prompts")
        return self


class BuildConfig(BaseModel):
    version: int = LLAMA_STACK_BUILD_CONFIG_VERSION

    distribution_spec: DistributionSpec = Field(description="The distribution spec to build including API providers. ")
    image_type: str = Field(
        default="venv",
        description="Type of package to build (container | venv)",
    )
    image_name: str | None = Field(
        default=None,
        description="Name of the distribution to build",
    )
    external_providers_dir: Path | None = Field(
        default=None,
        description="Path to directory containing external provider implementations. The providers packages will be resolved from this directory. "
        "pip_packages MUST contain the provider package name.",
    )
    additional_pip_packages: list[str] = Field(
        default_factory=list,
        description="Additional pip packages to install in the distribution. These packages will be installed in the distribution environment.",
    )
    external_apis_dir: Path | None = Field(
        default=None,
        description="Path to directory containing external API implementations. The APIs code and dependencies must be installed on the system.",
    )

    @field_validator("external_providers_dir")
    @classmethod
    def validate_external_providers_dir(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            return Path(v)
        return v

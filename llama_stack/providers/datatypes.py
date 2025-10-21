# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import StrEnum
from typing import Any, Protocol
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from llama_stack.apis.benchmarks import Benchmark
from llama_stack.apis.datasets import Dataset
from llama_stack.apis.datatypes import Api
from llama_stack.apis.models import Model
from llama_stack.apis.scoring_functions import ScoringFn
from llama_stack.apis.shields import Shield
from llama_stack.apis.tools import ToolGroup
from llama_stack.apis.vector_stores import VectorStore
from llama_stack.schema_utils import json_schema_type


class ModelsProtocolPrivate(Protocol):
    """
    Protocol for model management.

    This allows users to register their preferred model identifiers.

    Model registration requires -
     - a provider, used to route the registration request
     - a model identifier, user's intended name for the model during inference
     - a provider model identifier, a model identifier supported by the provider

    Providers will only accept registration for provider model ids they support.

    Example,
      register: provider x my-model-id x provider-model-id
       -> Error if provider does not support provider-model-id
       -> Error if my-model-id is already registered
       -> Success if provider supports provider-model-id
      inference: my-model-id x ...
       -> Provider uses provider-model-id for inference
    """

    # this should be called `on_model_register` or something like that.
    # the provider should _not_ be able to change the object in this
    # callback
    async def register_model(self, model: Model) -> Model: ...

    async def unregister_model(self, model_id: str) -> None: ...

    # the Stack router will query each provider for their list of models
    # if a `refresh_interval_seconds` is provided, this method will be called
    # periodically to refresh the list of models
    #
    # NOTE: each model returned will be registered with the model registry. this means
    # a callback to the `register_model()` method will be made. this is duplicative and
    # may be removed in the future.
    async def list_models(self) -> list[Model] | None: ...

    async def should_refresh_models(self) -> bool: ...


class ShieldsProtocolPrivate(Protocol):
    async def register_shield(self, shield: Shield) -> None: ...

    async def unregister_shield(self, identifier: str) -> None: ...


class VectorStoresProtocolPrivate(Protocol):
    async def register_vector_store(self, vector_store: VectorStore) -> None: ...

    async def unregister_vector_store(self, vector_store_id: str) -> None: ...


class DatasetsProtocolPrivate(Protocol):
    async def register_dataset(self, dataset: Dataset) -> None: ...

    async def unregister_dataset(self, dataset_id: str) -> None: ...


class ScoringFunctionsProtocolPrivate(Protocol):
    async def list_scoring_functions(self) -> list[ScoringFn]: ...

    async def register_scoring_function(self, scoring_fn: ScoringFn) -> None: ...


class BenchmarksProtocolPrivate(Protocol):
    async def register_benchmark(self, benchmark: Benchmark) -> None: ...


class ToolGroupsProtocolPrivate(Protocol):
    async def register_toolgroup(self, toolgroup: ToolGroup) -> None: ...

    async def unregister_toolgroup(self, toolgroup_id: str) -> None: ...


@json_schema_type
class ProviderSpec(BaseModel):
    api: Api
    provider_type: str
    config_class: str = Field(
        ...,
        description="Fully-qualified classname of the config for this provider",
    )
    api_dependencies: list[Api] = Field(
        default_factory=list,
        description="Higher-level API surfaces may depend on other providers to provide their functionality",
    )
    optional_api_dependencies: list[Api] = Field(
        default_factory=list,
    )
    deprecation_warning: str | None = Field(
        default=None,
        description="If this provider is deprecated, specify the warning message here",
    )
    deprecation_error: str | None = Field(
        default=None,
        description="If this provider is deprecated and does NOT work, specify the error message here",
    )

    module: str | None = Field(
        default=None,
        description="""
 Fully-qualified name of the module to import. The module is expected to have:

  - `get_adapter_impl(config, deps)`: returns the adapter implementation

  Example: `module: ramalama_stack`
 """,
    )

    pip_packages: list[str] = Field(
        default_factory=list,
        description="The pip dependencies needed for this implementation",
    )

    provider_data_validator: str | None = Field(
        default=None,
    )

    is_external: bool = Field(default=False, description="Notes whether this provider is an external provider.")

    # used internally by the resolver; this is a hack for now
    deps__: list[str] = Field(default_factory=list)

    @property
    def is_sample(self) -> bool:
        return self.provider_type in ("sample", "remote::sample")


class RoutingTable(Protocol):
    async def get_provider_impl(self, routing_key: str) -> Any: ...


@json_schema_type
class InlineProviderSpec(ProviderSpec):
    container_image: str | None = Field(
        default=None,
        description="""
The container image to use for this implementation. If one is provided, pip_packages will be ignored.
If a provider depends on other providers, the dependencies MUST NOT specify a container image.
""",
    )
    description: str | None = Field(
        default=None,
        description="""
A description of the provider. This is used to display in the documentation.
""",
    )


class RemoteProviderConfig(BaseModel):
    host: str = "localhost"
    port: int | None = None
    protocol: str = "http"

    @property
    def url(self) -> str:
        if self.port is None:
            return f"{self.protocol}://{self.host}"
        return f"{self.protocol}://{self.host}:{self.port}"

    @classmethod
    def from_url(cls, url: str) -> "RemoteProviderConfig":
        parsed = urlparse(url)
        attrs = {k: v for k, v in parsed._asdict().items() if v is not None}
        return cls(**attrs)


@json_schema_type
class RemoteProviderSpec(ProviderSpec):
    adapter_type: str = Field(
        ...,
        description="Unique identifier for this adapter",
    )

    description: str | None = Field(
        default=None,
        description="""
A description of the provider. This is used to display in the documentation.
""",
    )

    @property
    def container_image(self) -> str | None:
        return None


class HealthStatus(StrEnum):
    OK = "OK"
    ERROR = "Error"
    NOT_IMPLEMENTED = "Not Implemented"


HealthResponse = dict[str, Any]

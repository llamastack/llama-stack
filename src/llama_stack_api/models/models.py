# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pydantic models for Models API requests and responses.

This module defines the request and response models for the Models API
using Pydantic with Field descriptions for OpenAPI schema generation.
"""

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from llama_stack_api.resource import Resource, ResourceType
from llama_stack_api.schema_utils import json_schema_type


@json_schema_type
class ModelType(StrEnum):
    """Enumeration of supported model types in Llama Stack.

    :cvar llm: Large language model for text generation and completion
    :cvar embedding: Embedding model for converting text to vector representations
    :cvar rerank: Reranking model for reordering documents based on their relevance to a query
    """

    llm = "llm"
    embedding = "embedding"
    rerank = "rerank"


class CommonModelFields(BaseModel):
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional metadata for this model",
    )


@json_schema_type
class Model(CommonModelFields, Resource):
    """A model resource representing an AI model registered in Llama Stack.

    :param type: The resource type, always 'model' for model resources
    :param model_type: The type of model (LLM or embedding model)
    :param metadata: Any additional metadata for this model
    :param identifier: Unique identifier for this resource in llama stack
    :param provider_resource_id: Unique identifier for this resource in the provider
    :param provider_id: ID of the provider that owns this resource
    """

    type: Literal[ResourceType.model] = ResourceType.model

    @property
    def model_id(self) -> str:
        return self.identifier

    @property
    def provider_model_id(self) -> str:
        assert self.provider_resource_id is not None, "Provider resource ID must be set"
        return self.provider_resource_id

    model_config = ConfigDict(protected_namespaces=())

    model_type: ModelType = Field(default=ModelType.llm)

    @field_validator("provider_resource_id")
    @classmethod
    def validate_provider_resource_id(cls, v):
        if v is None:
            raise ValueError("provider_resource_id cannot be None")
        return v


class ModelInput(CommonModelFields):
    model_id: str
    provider_id: str | None = None
    provider_model_id: str | None = None
    model_type: ModelType | None = ModelType.llm
    model_config = ConfigDict(protected_namespaces=())


@json_schema_type
class ListModelsResponse(BaseModel):
    """Response containing a list of model objects."""

    data: list[Model] = Field(..., description="List of model objects.")


@json_schema_type
class OpenAIModel(BaseModel):
    """A model from OpenAI.

    :id: The ID of the model
    :object: The object type, which will be "model"
    :created: The Unix timestamp in seconds when the model was created
    :owned_by: The owner of the model
    :custom_metadata: Llama Stack-specific metadata including model_type, provider info, and additional metadata
    """

    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str
    custom_metadata: dict[str, Any] | None = None


@json_schema_type
class OpenAIListModelsResponse(BaseModel):
    """Response containing a list of OpenAI model objects."""

    data: list[OpenAIModel] = Field(..., description="List of OpenAI model objects.")


# Request models for each endpoint


@json_schema_type
class GetModelRequest(BaseModel):
    """Request model for getting a model by ID."""

    model_id: str = Field(..., description="The ID of the model to get.")


@json_schema_type
class RegisterModelRequest(BaseModel):
    """Request model for registering a model."""

    model_id: str = Field(..., description="The identifier of the model to register.")
    provider_model_id: str | None = Field(default=None, description="The identifier of the model in the provider.")
    provider_id: str | None = Field(default=None, description="The identifier of the provider.")
    metadata: dict[str, Any] | None = Field(default=None, description="Any additional metadata for this model.")
    model_type: ModelType | None = Field(default=None, description="The type of model to register.")


@json_schema_type
class UnregisterModelRequest(BaseModel):
    """Request model for unregistering a model."""

    model_id: str = Field(..., description="The ID of the model to unregister.")


__all__ = [
    "CommonModelFields",
    "GetModelRequest",
    "ListModelsResponse",
    "Model",
    "ModelInput",
    "ModelType",
    "OpenAIListModelsResponse",
    "OpenAIModel",
    "RegisterModelRequest",
    "UnregisterModelRequest",
]

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from llama_stack.apis.resource import Resource, ResourceType
from llama_stack.schema_utils import json_schema_type


class CommonModelFields(BaseModel):
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional metadata for this model.",
    )


@json_schema_type
class ModelType(StrEnum):
    """Enumeration of supported model types in Llama Stack."""

    llm = "llm"
    embedding = "embedding"
    rerank = "rerank"


@json_schema_type
class Model(CommonModelFields, Resource):
    """A model resource representing an AI model registered in Llama Stack."""

    type: Literal[ResourceType.model] = Field(
        default=ResourceType.model, description="The resource type, always 'model' for model resources."
    )
    model_type: ModelType = Field(default=ModelType.llm, description="The type of model (LLM or embedding model).")

    @property
    def model_id(self) -> str:
        return self.identifier

    @property
    def provider_model_id(self) -> str:
        assert self.provider_resource_id is not None, "Provider resource ID must be set"
        return self.provider_resource_id

    model_config = ConfigDict(protected_namespaces=())

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


class ListModelsResponse(BaseModel):
    """Response model for listing models."""

    data: list[Model] = Field(description="List of model resources.")


@json_schema_type
class RegisterModelRequest(BaseModel):
    """Request model for registering a new model."""

    model_id: str = Field(..., description="The identifier of the model to register.")
    provider_model_id: str | None = Field(default=None, description="The identifier of the model in the provider.")
    provider_id: str | None = Field(default=None, description="The identifier of the provider.")
    metadata: dict[str, Any] | None = Field(default=None, description="Any additional metadata for this model.")
    model_type: ModelType | None = Field(default=None, description="The type of model to register.")


@json_schema_type
class OpenAIModel(BaseModel):
    """A model from OpenAI."""

    id: str = Field(..., description="The ID of the model.")
    object: Literal["model"] = Field(default="model", description="The object type, which will be 'model'.")
    created: int = Field(..., description="The Unix timestamp in seconds when the model was created.")
    owned_by: str = Field(..., description="The owner of the model.")


class OpenAIListModelsResponse(BaseModel):
    """Response model for listing OpenAI models."""

    data: list[OpenAIModel] = Field(description="List of OpenAI model objects.")

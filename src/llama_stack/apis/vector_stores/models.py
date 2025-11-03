# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Literal

from pydantic import BaseModel, Field

from llama_stack.apis.resource import Resource, ResourceType


# Internal resource type for storing the vector store routing and other information
class VectorStore(Resource):
    """Vector database resource for storing and querying vector embeddings."""

    type: Literal[ResourceType.vector_store] = ResourceType.vector_store

    embedding_model: str = Field(..., description="Name of the embedding model to use for vector generation")
    embedding_dimension: int = Field(..., description="Dimension of the embedding vectors")
    vector_store_name: str | None = Field(default=None, description="Name of the vector store")

    @property
    def vector_store_id(self) -> str:
        return self.identifier

    @property
    def provider_vector_store_id(self) -> str | None:
        return self.provider_resource_id


class VectorStoreInput(BaseModel):
    """Input parameters for creating or configuring a vector database."""

    vector_store_id: str = Field(..., description="Unique identifier for the vector store")
    embedding_model: str = Field(..., description="Name of the embedding model to use for vector generation")
    embedding_dimension: int = Field(..., description="Dimension of the embedding vectors")
    provider_id: str | None = Field(default=None, description="ID of the provider that owns this vector store")
    provider_vector_store_id: str | None = Field(
        default=None, description="Provider-specific identifier for the vector store"
    )

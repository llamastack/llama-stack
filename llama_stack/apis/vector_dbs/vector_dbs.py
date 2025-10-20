# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from llama_stack.apis.resource import Resource, ResourceType
from llama_stack.schema_utils import json_schema_type


@json_schema_type
class VectorDB(Resource):
    """Vector database resource for storing and querying vector embeddings.

    :param type: Type of resource, always 'vector_db' for vector databases
    :param embedding_model: Name of the embedding model to use for vector generation
    :param embedding_dimension: Dimension of the embedding vectors
    """

    type: Literal[ResourceType.vector_db] = ResourceType.vector_db

    embedding_model: str
    embedding_dimension: int
    vector_db_name: str | None = None

    @property
    def vector_db_id(self) -> str:
        return self.identifier

    @property
    def provider_vector_db_id(self) -> str | None:
        return self.provider_resource_id


class VectorDBInput(BaseModel):
    """Input parameters for creating or configuring a vector database.

    :param vector_db_id: Unique identifier for the vector database
    :param embedding_model: Name of the embedding model to use for vector generation
    :param embedding_dimension: Dimension of the embedding vectors
    :param provider_vector_db_id: (Optional) Provider-specific identifier for the vector database
    """

    vector_db_id: str
    embedding_model: str
    embedding_dimension: int
    provider_id: str | None = None
    provider_vector_db_id: str | None = None


class ListVectorDBsResponse(BaseModel):
    """Response from listing vector databases.

    :param data: List of vector databases
    """

    data: list[VectorDB]


@runtime_checkable
class VectorDBs(Protocol):
    """Internal protocol for vector_dbs routing - no public API endpoints."""

    async def list_vector_dbs(self) -> ListVectorDBsResponse:
        """Internal method to list vector databases."""
        ...

    async def get_vector_db(
        self,
        vector_db_id: str,
    ) -> VectorDB:
        """Internal method to get a vector database by ID."""
        ...

    async def register_vector_db(
        self,
        vector_db_id: str,
        embedding_model: str,
        embedding_dimension: int | None = 384,
        provider_id: str | None = None,
        vector_db_name: str | None = None,
        provider_vector_db_id: str | None = None,
    ) -> VectorDB:
        """Internal method to register a vector database."""
        ...

    async def unregister_vector_db(self, vector_db_id: str) -> None:
        """Internal method to unregister a vector database."""
        ...

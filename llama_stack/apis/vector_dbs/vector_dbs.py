# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from llama_stack.apis.resource import Resource, ResourceType
from llama_stack.providers.utils.telemetry.trace_protocol import trace_protocol
from llama_stack.schema_utils import json_schema_type, webmethod


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
@trace_protocol
class VectorDBs(Protocol):
    @webmethod(route="/vector-dbs", method="GET")
    async def list_vector_dbs(self) -> ListVectorDBsResponse:
        """List all vector databases.

        :returns: A ListVectorDBsResponse.
        """
        ...

    @webmethod(route="/vector-dbs/{vector_db_id:path}", method="GET")
    async def get_vector_db(
        self,
        vector_db_id: str,
    ) -> VectorDB:
        """Get a vector database by its identifier.

        :param vector_db_id: The identifier of the vector database to get.
        :returns: A VectorDB.
        """
        ...

    @webmethod(route="/vector-dbs", method="POST")
    async def register_vector_db(
        self,
        vector_db_id: str,
        embedding_model: str,
        embedding_dimension: int | None = 384,
        provider_id: str | None = None,
        vector_db_name: str | None = None,
        provider_vector_db_id: str | None = None,
    ) -> VectorDB:
        """Register a vector database.

        :param vector_db_id: The identifier of the vector database to register.
        :param embedding_model: The embedding model to use.
        :param embedding_dimension: The dimension of the embedding model.
        :param provider_id: The identifier of the provider.
        :param vector_db_name: The name of the vector database.
        :param provider_vector_db_id: The identifier of the vector database in the provider.
        :returns: A VectorDB.
        """
        ...

    @webmethod(route="/vector-dbs/{vector_db_id:path}", method="DELETE")
    async def unregister_vector_db(self, vector_db_id: str) -> None:
        """Unregister a vector database.

        :param vector_db_id: The identifier of the vector database to unregister.
        """
        ...

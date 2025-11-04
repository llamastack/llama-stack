# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from llama_stack.core.storage.datatypes import KVStoreReference
from llama_stack.schema_utils import json_schema_type


@json_schema_type
class MongoDBVectorIOConfig(BaseModel):
    """Configuration for MongoDB Atlas Vector Search provider.

    This provider connects to MongoDB Atlas and uses Vector Search for RAG operations.
    """

    # MongoDB connection details - either connection_string or individual parameters
    connection_string: str | None = Field(
        default=None,
        description="MongoDB connection string (e.g., mongodb://user:pass@localhost:27017/ or mongodb+srv://user:pass@cluster.mongodb.net/)",
    )
    host: str | None = Field(
        default=None,
        description="MongoDB host (used if connection_string is not provided)",
    )
    port: int | None = Field(
        default=None,
        description="MongoDB port (used if connection_string is not provided)",
    )
    username: str | None = Field(
        default=None,
        description="MongoDB username (used if connection_string is not provided)",
    )
    password: str | None = Field(
        default=None,
        description="MongoDB password (used if connection_string is not provided)",
    )
    database_name: str = Field(default="llama_stack", description="Database name to use for vector collections")

    # Vector search configuration
    index_name: str = Field(default="vector_index", description="Name of the vector search index")
    path_field: str = Field(default="embedding", description="Field name for storing embeddings")
    similarity_metric: str = Field(
        default="cosine",
        description="Similarity metric: cosine, euclidean, or dotProduct",
    )

    # Connection options
    max_pool_size: int = Field(default=100, description="Maximum connection pool size")
    timeout_ms: int = Field(default=30000, description="Connection timeout in milliseconds")

    # KV store configuration
    persistence: KVStoreReference | None = Field(
        description="Config for KV store backend for metadata storage", default=None
    )

    def get_connection_string(self) -> str | None:
        """Build connection string from individual parameters if not provided directly.

        If both connection_string and individual parameters (host/port) are provided,
        individual parameters take precedence to allow test environment overrides.
        """
        # Prioritize individual connection parameters over connection_string
        # This allows test environments to override with MONGODB_HOST/PORT/etc
        if self.host and self.port:
            auth_part = ""
            if self.username and self.password:
                auth_part = f"{self.username}:{self.password}@"
            return f"mongodb://{auth_part}{self.host}:{self.port}/"

        # Fall back to connection_string if provided
        if self.connection_string:
            return self.connection_string

        return None

    @classmethod
    def sample_run_config(
        cls,
        __distro_dir__: str,
        connection_string: str = "${env.MONGODB_CONNECTION_STRING:=}",
        host: str = "${env.MONGODB_HOST:=localhost}",
        port: str = "${env.MONGODB_PORT:=27017}",
        username: str = "${env.MONGODB_USERNAME:=}",
        password: str = "${env.MONGODB_PASSWORD:=}",
        database_name: str = "${env.MONGODB_DATABASE_NAME:=llama_stack}",
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "connection_string": connection_string,
            "host": host,
            "port": port,
            "username": username,
            "password": password,
            "database_name": database_name,
            "index_name": "${env.MONGODB_INDEX_NAME:=vector_index}",
            "path_field": "${env.MONGODB_PATH_FIELD:=embedding}",
            "similarity_metric": "${env.MONGODB_SIMILARITY_METRIC:=cosine}",
            "max_pool_size": "${env.MONGODB_MAX_POOL_SIZE:=100}",
            "timeout_ms": "${env.MONGODB_TIMEOUT_MS:=30000}",
            "persistence": KVStoreReference(
                backend="kv_default",
                namespace="vector_io::mongodb_atlas",
            ).model_dump(exclude_none=True),
        }

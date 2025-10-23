# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import re
from abc import abstractmethod
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator


class StorageBackendType(StrEnum):
    KV_REDIS = "kv_redis"
    KV_SQLITE = "kv_sqlite"
    KV_POSTGRES = "kv_postgres"
    KV_MONGODB = "kv_mongodb"
    SQL_SQLITE = "sql_sqlite"
    SQL_POSTGRES = "sql_postgres"


class CommonConfig(BaseModel):
    namespace: str | None = Field(
        default=None,
        description="All keys will be prefixed with this namespace",
    )


class RedisKVStoreConfig(CommonConfig):
    type: Literal[StorageBackendType.KV_REDIS] = StorageBackendType.KV_REDIS
    host: str = "localhost"
    port: int = 6379

    @property
    def url(self) -> str:
        return f"redis://{self.host}:{self.port}"

    @classmethod
    def pip_packages(cls) -> list[str]:
        return ["redis"]

    @classmethod
    def sample_run_config(cls):
        return {
            "type": StorageBackendType.KV_REDIS.value,
            "host": "${env.REDIS_HOST:=localhost}",
            "port": "${env.REDIS_PORT:=6379}",
        }


class SqliteKVStoreConfig(CommonConfig):
    type: Literal[StorageBackendType.KV_SQLITE] = StorageBackendType.KV_SQLITE
    db_path: str = Field(
        description="File path for the sqlite database",
    )

    @classmethod
    def pip_packages(cls) -> list[str]:
        return ["aiosqlite"]

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, db_name: str = "kvstore.db"):
        return {
            "type": StorageBackendType.KV_SQLITE.value,
            "db_path": "${env.SQLITE_STORE_DIR:=" + __distro_dir__ + "}/" + db_name,
        }


class PostgresKVStoreConfig(CommonConfig):
    type: Literal[StorageBackendType.KV_POSTGRES] = StorageBackendType.KV_POSTGRES
    host: str = "localhost"
    port: int | str = 5432
    db: str = "llamastack"
    user: str
    password: str | None = None
    ssl_mode: str | None = None
    ca_cert_path: str | None = None
    table_name: str = "llamastack_kvstore"

    @classmethod
    def sample_run_config(cls, table_name: str = "llamastack_kvstore", **kwargs):
        return {
            "type": StorageBackendType.KV_POSTGRES.value,
            "host": "${env.POSTGRES_HOST:=localhost}",
            "port": "${env.POSTGRES_PORT:=5432}",
            "db": "${env.POSTGRES_DB:=llamastack}",
            "user": "${env.POSTGRES_USER:=llamastack}",
            "password": "${env.POSTGRES_PASSWORD:=llamastack}",
            "table_name": "${env.POSTGRES_TABLE_NAME:=" + table_name + "}",
        }

    @classmethod
    @field_validator("table_name")
    def validate_table_name(cls, v: str) -> str:
        # PostgreSQL identifiers rules:
        # - Must start with a letter or underscore
        # - Can contain letters, numbers, and underscores
        # - Maximum length is 63 bytes
        pattern = r"^[a-zA-Z_][a-zA-Z0-9_]*$"
        if not re.match(pattern, v):
            raise ValueError(
                "Invalid table name. Must start with letter or underscore and contain only letters, numbers, and underscores"
            )
        if len(v) > 63:
            raise ValueError("Table name must be less than 63 characters")
        return v

    @classmethod
    def pip_packages(cls) -> list[str]:
        return ["psycopg2-binary"]


class MongoDBKVStoreConfig(CommonConfig):
    type: Literal[StorageBackendType.KV_MONGODB] = StorageBackendType.KV_MONGODB
    host: str = "localhost"
    port: int = 27017
    db: str = "llamastack"
    user: str | None = None
    password: str | None = None
    collection_name: str = "llamastack_kvstore"

    @classmethod
    def pip_packages(cls) -> list[str]:
        return ["pymongo"]

    @classmethod
    def sample_run_config(cls, collection_name: str = "llamastack_kvstore"):
        return {
            "type": StorageBackendType.KV_MONGODB.value,
            "host": "${env.MONGODB_HOST:=localhost}",
            "port": "${env.MONGODB_PORT:=5432}",
            "db": "${env.MONGODB_DB}",
            "user": "${env.MONGODB_USER}",
            "password": "${env.MONGODB_PASSWORD}",
            "collection_name": "${env.MONGODB_COLLECTION_NAME:=" + collection_name + "}",
        }


class SqlAlchemySqlStoreConfig(BaseModel):
    @property
    @abstractmethod
    def engine_str(self) -> str: ...

    # TODO: move this when we have a better way to specify dependencies with internal APIs
    @classmethod
    def pip_packages(cls) -> list[str]:
        return ["sqlalchemy[asyncio]"]


class SqliteSqlStoreConfig(SqlAlchemySqlStoreConfig):
    type: Literal[StorageBackendType.SQL_SQLITE] = StorageBackendType.SQL_SQLITE
    db_path: str = Field(
        description="Database path, e.g. ~/.llama/distributions/ollama/sqlstore.db",
    )

    @property
    def engine_str(self) -> str:
        return "sqlite+aiosqlite:///" + Path(self.db_path).expanduser().as_posix()

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, db_name: str = "sqlstore.db"):
        return {
            "type": StorageBackendType.SQL_SQLITE.value,
            "db_path": "${env.SQLITE_STORE_DIR:=" + __distro_dir__ + "}/" + db_name,
        }

    @classmethod
    def pip_packages(cls) -> list[str]:
        return super().pip_packages() + ["aiosqlite"]


class PostgresSqlStoreConfig(SqlAlchemySqlStoreConfig):
    type: Literal[StorageBackendType.SQL_POSTGRES] = StorageBackendType.SQL_POSTGRES
    host: str = "localhost"
    port: int | str = 5432
    db: str = "llamastack"
    user: str
    password: str | None = None

    @property
    def engine_str(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"

    @classmethod
    def pip_packages(cls) -> list[str]:
        return super().pip_packages() + ["asyncpg"]

    @classmethod
    def sample_run_config(cls, **kwargs):
        return {
            "type": StorageBackendType.SQL_POSTGRES.value,
            "host": "${env.POSTGRES_HOST:=localhost}",
            "port": "${env.POSTGRES_PORT:=5432}",
            "db": "${env.POSTGRES_DB:=llamastack}",
            "user": "${env.POSTGRES_USER:=llamastack}",
            "password": "${env.POSTGRES_PASSWORD:=llamastack}",
        }


# reference = (backend_name, table_name)
class SqlStoreReference(BaseModel):
    """A reference to a 'SQL-like' persistent store. A table name must be provided."""

    table_name: str = Field(
        description="Name of the table to use for the SqlStore",
    )

    backend: str = Field(
        description="Name of backend from storage.backends",
    )


# reference = (backend_name, namespace)
class KVStoreReference(BaseModel):
    """A reference to a 'key-value' persistent store. A namespace must be provided."""

    namespace: str = Field(
        description="Key prefix for KVStore backends",
    )

    backend: str = Field(
        description="Name of backend from storage.backends",
    )


StorageBackendConfig = Annotated[
    RedisKVStoreConfig
    | SqliteKVStoreConfig
    | PostgresKVStoreConfig
    | MongoDBKVStoreConfig
    | SqliteSqlStoreConfig
    | PostgresSqlStoreConfig,
    Field(discriminator="type"),
]


class InferenceStoreReference(SqlStoreReference):
    """Inference store configuration with queue tuning."""

    max_write_queue_size: int = Field(
        default=10000,
        description="Max queued writes for inference store",
    )
    num_writers: int = Field(
        default=4,
        description="Number of concurrent background writers",
    )


class ResponsesStoreReference(InferenceStoreReference):
    """Responses store configuration with queue tuning."""


class ServerStoresConfig(BaseModel):
    metadata: KVStoreReference | None = Field(
        default=None,
        description="Metadata store configuration (uses KV backend)",
    )
    inference: InferenceStoreReference | None = Field(
        default=None,
        description="Inference store configuration (uses SQL backend)",
    )
    conversations: SqlStoreReference | None = Field(
        default=None,
        description="Conversations store configuration (uses SQL backend)",
    )
    responses: ResponsesStoreReference | None = Field(
        default=None,
        description="Responses store configuration (uses SQL backend)",
    )
    prompts: KVStoreReference | None = Field(
        default=None,
        description="Prompts store configuration (uses KV backend)",
    )


class StorageConfig(BaseModel):
    backends: dict[str, StorageBackendConfig] = Field(
        description="Named backend configurations (e.g., 'default', 'cache')",
    )
    stores: ServerStoresConfig = Field(
        default_factory=lambda: ServerStoresConfig(),
        description="Named references to storage backends used by the stack core",
    )

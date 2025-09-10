# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import re
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

from llama_stack.core.utils.config_dirs import RUNTIME_BASE_DIR


class KVStoreType(Enum):
    redis = "redis"
    sqlite = "sqlite"
    postgres = "postgres"
    mongodb = "mongodb"


class CommonConfig(BaseModel):
    namespace: str | None = Field(
        default=None,
        description="All keys will be prefixed with this namespace",
    )


class RedisKVStoreConfig(CommonConfig):
    type: Literal[KVStoreType.redis.value] = KVStoreType.redis.value
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
            "type": "redis",
            "host": "${env.REDIS_HOST:=localhost}",
            "port": "${env.REDIS_PORT:=6379}",
        }


class SqliteKVStoreConfig(CommonConfig):
    type: Literal[KVStoreType.sqlite.value] = KVStoreType.sqlite.value
    db_path: str = Field(
        default=(RUNTIME_BASE_DIR / "kvstore.db").as_posix(),
        description="File path for the sqlite database",
    )

    @classmethod
    def pip_packages(cls) -> list[str]:
        return ["aiosqlite"]

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, db_name: str = "kvstore.db"):
        return {
            "type": "sqlite",
            "db_path": "${env.SQLITE_STORE_DIR:=" + __distro_dir__ + "}/" + db_name,
        }


class PostgresKVStoreConfig(CommonConfig):
    type: Literal[KVStoreType.postgres.value] = KVStoreType.postgres.value
    host: str = "localhost"
    port: int = 5432
    db: str = "llamastack"
    user: str
    password: str | None = None
    ssl_mode: str | None = None
    ca_cert_path: str | None = None
    table_name: str = "llamastack_kvstore"

    @classmethod
    def sample_run_config(cls, table_name: str = "llamastack_kvstore", **kwargs):
        return {
            "type": "postgres",
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
    type: Literal[KVStoreType.mongodb.value] = KVStoreType.mongodb.value
    host: str = "localhost"
    port: int = 27017
    db: str = "llamastack"
    user: str = None
    password: str | None = None
    collection_name: str = "llamastack_kvstore"

    @classmethod
    def pip_packages(cls) -> list[str]:
        return ["pymongo"]

    @classmethod
    def sample_run_config(cls, collection_name: str = "llamastack_kvstore"):
        return {
            "type": "mongodb",
            "host": "${env.MONGODB_HOST:=localhost}",
            "port": "${env.MONGODB_PORT:=5432}",
            "db": "${env.MONGODB_DB}",
            "user": "${env.MONGODB_USER}",
            "password": "${env.MONGODB_PASSWORD}",
            "collection_name": "${env.MONGODB_COLLECTION_NAME:=" + collection_name + "}",
        }


KVStoreConfig = Annotated[
    RedisKVStoreConfig | SqliteKVStoreConfig | PostgresKVStoreConfig | MongoDBKVStoreConfig,
    Field(discriminator="type", default=KVStoreType.sqlite.value),
]


def get_pip_packages(store_config: dict | KVStoreConfig) -> list[str]:
    """Get pip packages for KV store config, handling both dict and object cases."""
    if isinstance(store_config, dict):
        store_type = store_config.get("type")
        if store_type == "sqlite":
            return SqliteKVStoreConfig.pip_packages()
        elif store_type == "postgres":
            return PostgresKVStoreConfig.pip_packages()
        elif store_type == "redis":
            return RedisKVStoreConfig.pip_packages()
        elif store_type == "mongodb":
            return MongoDBKVStoreConfig.pip_packages()
        else:
            raise ValueError(f"Unknown KV store type: {store_type}")
    else:
        return store_config.pip_packages()

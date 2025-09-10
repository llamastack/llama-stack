# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from abc import abstractmethod
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from llama_stack.core.utils.config_dirs import RUNTIME_BASE_DIR

from .api import SqlStore

sql_store_pip_packages = ["sqlalchemy[asyncio]", "aiosqlite", "asyncpg"]


class SqlStoreType(StrEnum):
    sqlite = "sqlite"
    postgres = "postgres"


class SqlAlchemySqlStoreConfig(BaseModel):
    @property
    @abstractmethod
    def engine_str(self) -> str: ...

    # TODO: move this when we have a better way to specify dependencies with internal APIs
    @classmethod
    def pip_packages(cls) -> list[str]:
        return ["sqlalchemy[asyncio]"]


class SqliteSqlStoreConfig(SqlAlchemySqlStoreConfig):
    type: Literal[SqlStoreType.sqlite] = SqlStoreType.sqlite
    db_path: str = Field(
        default=(RUNTIME_BASE_DIR / "sqlstore.db").as_posix(),
        description="Database path, e.g. ~/.llama/distributions/ollama/sqlstore.db",
    )

    @property
    def engine_str(self) -> str:
        return "sqlite+aiosqlite:///" + Path(self.db_path).expanduser().as_posix()

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, db_name: str = "sqlstore.db"):
        return {
            "type": "sqlite",
            "db_path": "${env.SQLITE_STORE_DIR:=" + __distro_dir__ + "}/" + db_name,
        }

    @classmethod
    def pip_packages(cls) -> list[str]:
        return super().pip_packages() + ["aiosqlite"]


class PostgresSqlStoreConfig(SqlAlchemySqlStoreConfig):
    type: Literal[SqlStoreType.postgres] = SqlStoreType.postgres
    host: str = "localhost"
    port: int = 5432
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
            "type": "postgres",
            "host": "${env.POSTGRES_HOST:=localhost}",
            "port": "${env.POSTGRES_PORT:=5432}",
            "db": "${env.POSTGRES_DB:=llamastack}",
            "user": "${env.POSTGRES_USER:=llamastack}",
            "password": "${env.POSTGRES_PASSWORD:=llamastack}",
        }


SqlStoreConfig = Annotated[
    SqliteSqlStoreConfig | PostgresSqlStoreConfig,
    Field(discriminator="type", default=SqlStoreType.sqlite.value),
]


def get_pip_packages(store_config: dict | SqlStoreConfig) -> list[str]:
    """Get pip packages for SQL store config, handling both dict and object cases."""
    if isinstance(store_config, dict):
        store_type = store_config.get("type")
        if store_type == "sqlite":
            return SqliteSqlStoreConfig.pip_packages()
        elif store_type == "postgres":
            return PostgresSqlStoreConfig.pip_packages()
        else:
            raise ValueError(f"Unknown SQL store type: {store_type}")
    else:
        return store_config.pip_packages()


def sqlstore_impl(config: SqlStoreConfig) -> SqlStore:
    if config.type in [SqlStoreType.sqlite, SqlStoreType.postgres]:
        from .sqlalchemy_sqlstore import SqlAlchemySqlStoreImpl

        impl = SqlAlchemySqlStoreImpl(config)
    else:
        raise ValueError(f"Unknown sqlstore type {config.type}")

    return impl

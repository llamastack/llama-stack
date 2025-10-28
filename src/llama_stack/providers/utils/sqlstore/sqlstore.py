# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated, cast

from pydantic import Field

from llama_stack.core.storage.datatypes import (
    PostgresSqlStoreConfig,
    SqliteSqlStoreConfig,
    SqlStoreReference,
    StorageBackendConfig,
    StorageBackendType,
)

from .api import SqlStore

sql_store_pip_packages = ["sqlalchemy[asyncio]", "aiosqlite", "asyncpg"]

_SQLSTORE_BACKENDS: dict[str, StorageBackendConfig] = {}


SqlStoreConfig = Annotated[
    SqliteSqlStoreConfig | PostgresSqlStoreConfig,
    Field(discriminator="type"),
]


def get_pip_packages(store_config: dict | SqlStoreConfig) -> list[str]:
    """Get pip packages for SQL store config, handling both dict and object cases."""
    if isinstance(store_config, dict):
        store_type = store_config.get("type")
        if store_type == StorageBackendType.SQL_SQLITE.value:
            return SqliteSqlStoreConfig.pip_packages()
        elif store_type == StorageBackendType.SQL_POSTGRES.value:
            return PostgresSqlStoreConfig.pip_packages()
        else:
            raise ValueError(f"Unknown SQL store type: {store_type}")
    else:
        return store_config.pip_packages()


def sqlstore_impl(reference: SqlStoreReference) -> SqlStore:
    backend_name = reference.backend

    backend_config = _SQLSTORE_BACKENDS.get(backend_name)
    if backend_config is None:
        raise ValueError(
            f"Unknown SQL store backend '{backend_name}'. Registered backends: {sorted(_SQLSTORE_BACKENDS)}"
        )

    if isinstance(backend_config, SqliteSqlStoreConfig | PostgresSqlStoreConfig):
        from .sqlalchemy_sqlstore import SqlAlchemySqlStoreImpl

        config = cast(SqliteSqlStoreConfig | PostgresSqlStoreConfig, backend_config).model_copy()
        return SqlAlchemySqlStoreImpl(config)
    else:
        raise ValueError(f"Unknown sqlstore type {backend_config.type}")


def register_sqlstore_backends(backends: dict[str, StorageBackendConfig]) -> None:
    """Register the set of available SQL store backends for reference resolution."""
    global _SQLSTORE_BACKENDS

    _SQLSTORE_BACKENDS.clear()
    for name, cfg in backends.items():
        _SQLSTORE_BACKENDS[name] = cfg

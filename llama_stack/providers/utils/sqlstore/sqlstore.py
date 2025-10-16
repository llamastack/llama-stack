# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated

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
_SQLSTORE_DEFAULT_BACKEND: str | None = None


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
    backend_name = reference.backend or _SQLSTORE_DEFAULT_BACKEND
    if not backend_name:
        raise ValueError(
            "SQL store reference did not specify a backend and no default backend is configured. "
            f"Available backends: {sorted(_SQLSTORE_BACKENDS)}"
        )

    backend_config = _SQLSTORE_BACKENDS.get(backend_name)
    if backend_config.type in [StorageBackendType.SQL_SQLITE, StorageBackendType.SQL_POSTGRES]:
        from .sqlalchemy_sqlstore import SqlAlchemySqlStoreImpl

        config = backend_config.model_copy()
        config.table_name = reference.table_name
        return SqlAlchemySqlStoreImpl(config)
    else:
        raise ValueError(f"Unknown sqlstore type {backend_config.type}")


def register_sqlstore_backends(backends: dict[str, StorageBackendConfig]) -> None:
    """Register the set of available SQL store backends for reference resolution."""
    global _SQLSTORE_BACKENDS

    def _set_default_backend(name: str) -> None:
        global _SQLSTORE_DEFAULT_BACKEND
        if _SQLSTORE_DEFAULT_BACKEND and _SQLSTORE_DEFAULT_BACKEND != name:
            raise ValueError(
                f"Multiple SQL store backends marked as default: '{_SQLSTORE_DEFAULT_BACKEND}' and '{name}'. "
                "Only one backend can be the default."
            )
        _SQLSTORE_DEFAULT_BACKEND = name

    _SQLSTORE_BACKENDS.clear()
    for name, cfg in backends.items():
        if cfg.default:
            _set_default_backend(name)

        _SQLSTORE_BACKENDS[name] = cfg

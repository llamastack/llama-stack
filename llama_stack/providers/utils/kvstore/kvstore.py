# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

from llama_stack.core.storage.datatypes import KVStoreReference, StorageBackendConfig, StorageBackendType

from .api import KVStore
from .config import KVStoreConfig


def kvstore_dependencies():
    """
    Returns all possible kvstore dependencies for registry/provider specifications.

    NOTE: For specific kvstore implementations, use config.pip_packages instead.
    This function returns the union of all dependencies for cases where the specific
    kvstore type is not known at declaration time (e.g., provider registries).
    """
    return ["aiosqlite", "psycopg2-binary", "redis", "pymongo"]


class InmemoryKVStoreImpl(KVStore):
    def __init__(self):
        self._store = {}

    async def initialize(self) -> None:
        pass

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def set(self, key: str, value: str) -> None:
        self._store[key] = value

    async def values_in_range(self, start_key: str, end_key: str) -> list[str]:
        return [self._store[key] for key in self._store.keys() if key >= start_key and key < end_key]

    async def keys_in_range(self, start_key: str, end_key: str) -> list[str]:
        """Get all keys in the given range."""
        return [key for key in self._store.keys() if key >= start_key and key < end_key]

    async def delete(self, key: str) -> None:
        del self._store[key]


_KVSTORE_BACKENDS: dict[str, KVStoreConfig] = {}
_KVSTORE_DEFAULT_BACKEND: str | None = None


def register_kvstore_backends(backends: dict[str, StorageBackendConfig]) -> None:
    """Register the set of available KV store backends for reference resolution."""
    global _KVSTORE_BACKENDS

    def _set_default_backend(name: str) -> None:
        global _KVSTORE_DEFAULT_BACKEND
        if _KVSTORE_DEFAULT_BACKEND and _KVSTORE_DEFAULT_BACKEND != name:
            raise ValueError(
                f"Multiple KVStore backends marked as default: '{_KVSTORE_DEFAULT_BACKEND}' and '{name}'. "
                "Only one backend can be the default."
            )
        _KVSTORE_DEFAULT_BACKEND = name

    _KVSTORE_BACKENDS.clear()
    for name, cfg in backends.items():
        if cfg.default:
            _set_default_backend(name)
        _KVSTORE_BACKENDS[name] = cfg


async def kvstore_impl(reference: KVStoreReference) -> KVStore:
    backend_name = reference.backend or _KVSTORE_DEFAULT_BACKEND
    if not backend_name:
        raise ValueError(
            "KVStore reference did not specify a backend and no default backend is configured. "
            f"Available backends: {sorted(_KVSTORE_BACKENDS)}"
        )

    backend_config = _KVSTORE_BACKENDS.get(backend_name)
    if backend_config is None:
        raise ValueError(f"Unknown KVStore backend '{backend_name}'. Registered backends: {sorted(_KVSTORE_BACKENDS)}")

    config = backend_config.model_copy()
    config.namespace = reference.namespace

    if config.type == StorageBackendType.KV_REDIS.value:
        from .redis import RedisKVStoreImpl

        impl = RedisKVStoreImpl(config)
    elif config.type == StorageBackendType.KV_SQLITE.value:
        from .sqlite import SqliteKVStoreImpl

        impl = SqliteKVStoreImpl(config)
    elif config.type == StorageBackendType.KV_POSTGRES.value:
        from .postgres import PostgresKVStoreImpl

        impl = PostgresKVStoreImpl(config)
    elif config.type == StorageBackendType.KV_MONGODB.value:
        from .mongodb import MongoDBKVStoreImpl

        impl = MongoDBKVStoreImpl(config)
    else:
        raise ValueError(f"Unknown kvstore type {config.type}")

    await impl.initialize()
    return impl

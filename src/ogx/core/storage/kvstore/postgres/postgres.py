# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime

import asyncpg

from ogx.log import get_logger
from ogx_api.internal.kvstore import KVStore

from ..config import PostgresKVStoreConfig

log = get_logger(name=__name__, category="providers::utils")


class PostgresKVStoreImpl(KVStore):
    """PostgreSQL-backed key-value store implementation."""

    def __init__(self, config: PostgresKVStoreConfig):
        self.config = config
        self._pool: asyncpg.Pool | None = None

    async def initialize(self) -> None:
        try:
            ssl: object = None
            if self.config.ssl_mode == "verify-full" and self.config.ca_cert_path:
                import ssl as _ssl

                ssl = _ssl.create_default_context(cafile=self.config.ca_cert_path)
            elif self.config.ssl_mode and self.config.ssl_mode != "disable":
                ssl = self.config.ssl_mode

            self._pool = await asyncpg.create_pool(
                host=self.config.host,
                port=int(self.config.port),
                database=self.config.db,
                user=self.config.user,
                password=self.config.password,
                ssl=ssl,
                min_size=1,
                max_size=10,
            )

            async with self._pool.acquire() as conn:
                await conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.config.table_name} (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        expiration TIMESTAMP
                    )
                    """
                )
        except Exception as e:
            log.exception("Could not connect to PostgreSQL database server")
            raise RuntimeError("Could not connect to PostgreSQL database server") from e

    def _pool_or_raise(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("Postgres client not initialized")
        return self._pool

    def _namespaced_key(self, key: str) -> str:
        if not self.config.namespace:
            return key
        return f"{self.config.namespace}:{key}"

    def _strip_namespace(self, key: str) -> str:
        if self.config.namespace and key.startswith(f"{self.config.namespace}:"):
            return key[len(self.config.namespace) + 1 :]
        return key

    async def set(self, key: str, value: str, expiration: datetime | None = None) -> None:
        key = self._namespaced_key(key)
        pool = self._pool_or_raise()
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self.config.table_name} (key, value, expiration)
                VALUES ($1, $2, $3)
                ON CONFLICT (key) DO UPDATE
                SET value = EXCLUDED.value, expiration = EXCLUDED.expiration
                """,
                key,
                value,
                expiration,
            )

    async def get(self, key: str) -> str | None:
        key = self._namespaced_key(key)
        pool = self._pool_or_raise()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT value FROM {self.config.table_name}
                WHERE key = $1
                AND (expiration IS NULL OR expiration > NOW())
                """,
                key,
            )
            return row["value"] if row else None

    async def delete(self, key: str) -> None:
        key = self._namespaced_key(key)
        pool = self._pool_or_raise()
        async with pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {self.config.table_name} WHERE key = $1",
                key,
            )

    async def values_in_range(self, start_key: str, end_key: str) -> list[str]:
        start_key = self._namespaced_key(start_key)
        end_key = self._namespaced_key(end_key)

        pool = self._pool_or_raise()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT value FROM {self.config.table_name}
                WHERE key >= $1 AND key < $2
                AND (expiration IS NULL OR expiration > NOW())
                ORDER BY key
                """,
                start_key,
                end_key,
            )
            return [row["value"] for row in rows]

    async def keys_in_range(self, start_key: str, end_key: str) -> list[str]:
        start_key = self._namespaced_key(start_key)
        end_key = self._namespaced_key(end_key)

        pool = self._pool_or_raise()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT key FROM {self.config.table_name}
                WHERE key >= $1 AND key < $2
                AND (expiration IS NULL OR expiration > NOW())
                ORDER BY key
                """,
                start_key,
                end_key,
            )
            return [self._strip_namespace(row["key"]) for row in rows]

    async def shutdown(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

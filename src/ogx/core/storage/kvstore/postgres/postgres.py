# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import Callable
from datetime import datetime
from typing import TypeVar

import psycopg2  # type: ignore[import-not-found]
from psycopg2.extensions import connection as PGConnection  # type: ignore[import-not-found]
from psycopg2.extras import DictCursor  # type: ignore[import-not-found]

from ogx.log import get_logger
from ogx_api.internal.kvstore import KVStore

from ..config import PostgresKVStoreConfig

log = get_logger(name=__name__, category="providers::utils")

T = TypeVar("T")


class PostgresKVStoreImpl(KVStore):
    """PostgreSQL-backed key-value store implementation."""

    def __init__(self, config: PostgresKVStoreConfig):
        self.config = config
        self._conn: PGConnection | None = None
        self._cursor: DictCursor | None = None

    def _connect(self) -> None:
        self._conn = psycopg2.connect(
            host=self.config.host,
            port=self.config.port,
            database=self.config.db,
            user=self.config.user,
            password=self.config.password,
            sslmode=self.config.ssl_mode,
            sslrootcert=self.config.ca_cert_path,
        )
        self._conn.autocommit = True
        self._cursor = self._conn.cursor(cursor_factory=DictCursor)

    async def initialize(self) -> None:
        try:
            self._connect()
            self._cursor_or_raise().execute(
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

    def _reconnect(self) -> None:
        """Close stale connection and establish a new one."""
        try:
            if self._cursor:
                self._cursor.close()
        except Exception:
            pass
        try:
            if self._conn:
                self._conn.close()
        except Exception:
            pass
        self._conn = None
        self._cursor = None
        self._connect()

    def _execute_with_retry(self, fn: Callable[[], T]) -> T:
        """Execute fn, reconnecting once on connection errors."""
        try:
            return fn()
        except (psycopg2.InterfaceError, psycopg2.OperationalError):
            log.warning("PostgreSQL connection lost, reconnecting")
            self._reconnect()
            return fn()

    def _cursor_or_raise(self) -> DictCursor:
        if self._cursor is None:
            raise RuntimeError("Postgres client not initialized")
        return self._cursor

    def _namespaced_key(self, key: str) -> str:
        if not self.config.namespace:
            return key
        return f"{self.config.namespace}:{key}"

    async def set(self, key: str, value: str, expiration: datetime | None = None) -> None:
        key = self._namespaced_key(key)

        def _do() -> None:
            self._cursor_or_raise().execute(
                f"""
                INSERT INTO {self.config.table_name} (key, value, expiration)
                VALUES (%s, %s, %s)
                ON CONFLICT (key) DO UPDATE
                SET value = EXCLUDED.value, expiration = EXCLUDED.expiration
                """,
                (key, value, expiration),
            )

        self._execute_with_retry(_do)

    async def get(self, key: str) -> str | None:
        key = self._namespaced_key(key)

        def _do() -> str | None:
            cursor = self._cursor_or_raise()
            cursor.execute(
                f"""
                SELECT value FROM {self.config.table_name}
                WHERE key = %s
                AND (expiration IS NULL OR expiration > NOW())
                """,
                (key,),
            )
            result = cursor.fetchone()
            return result[0] if result else None

        return self._execute_with_retry(_do)

    async def delete(self, key: str) -> None:
        key = self._namespaced_key(key)

        def _do() -> None:
            self._cursor_or_raise().execute(
                f"DELETE FROM {self.config.table_name} WHERE key = %s",
                (key,),
            )

        self._execute_with_retry(_do)

    async def values_in_range(self, start_key: str, end_key: str) -> list[str]:
        start_key = self._namespaced_key(start_key)
        end_key = self._namespaced_key(end_key)

        def _do() -> list[str]:
            cursor = self._cursor_or_raise()
            cursor.execute(
                f"""
                SELECT value FROM {self.config.table_name}
                WHERE key >= %s AND key < %s
                AND (expiration IS NULL OR expiration > NOW())
                ORDER BY key
                """,
                (start_key, end_key),
            )
            return [row[0] for row in cursor.fetchall()]

        return self._execute_with_retry(_do)

    async def keys_in_range(self, start_key: str, end_key: str) -> list[str]:
        start_key = self._namespaced_key(start_key)
        end_key = self._namespaced_key(end_key)

        def _do() -> list[str]:
            cursor = self._cursor_or_raise()
            cursor.execute(
                f"""
                SELECT key FROM {self.config.table_name}
                WHERE key >= %s AND key < %s
                AND (expiration IS NULL OR expiration > NOW())
                ORDER BY key
                """,
                (start_key, end_key),
            )
            return [row[0] for row in cursor.fetchall()]

        return self._execute_with_retry(_do)

    async def shutdown(self) -> None:
        if self._cursor:
            self._cursor.close()
            self._cursor = None
        if self._conn:
            self._conn.close()
            self._conn = None

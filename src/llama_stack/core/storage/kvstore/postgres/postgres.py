# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime

import psycopg2
from psycopg2.extras import DictCursor

from llama_stack.log import get_logger

from ..api import KVStore
from ..config import PostgresKVStoreConfig

log = get_logger(name=__name__, category="providers::utils")


class PostgresKVStoreImpl(KVStore):
    def __init__(self, config: PostgresKVStoreConfig):
        self.config = config
        self.conn = None
        self.cursor = None

    async def initialize(self) -> None:
        try:
            self.conn = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.db,
                user=self.config.user,
                password=self.config.password,
                sslmode=self.config.ssl_mode,
                sslrootcert=self.config.ca_cert_path,
            )
            self.conn.autocommit = True
            self.cursor = self.conn.cursor(cursor_factory=DictCursor)

            # Create table if it doesn't exist
            self.cursor.execute(
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

    def _namespaced_key(self, key: str) -> str:
        if not self.config.namespace:
            return key
        return f"{self.config.namespace}:{key}"

    async def set(self, key: str, value: str, expiration: datetime | None = None) -> None:
        key = self._namespaced_key(key)
        self.cursor.execute(
            f"""
            INSERT INTO {self.config.table_name} (key, value, expiration)
            VALUES (%s, %s, %s)
            ON CONFLICT (key) DO UPDATE
            SET value = EXCLUDED.value, expiration = EXCLUDED.expiration
            """,
            (key, value, expiration),
        )

    async def get(self, key: str) -> str | None:
        key = self._namespaced_key(key)
        self.cursor.execute(
            f"""
            SELECT value FROM {self.config.table_name}
            WHERE key = %s
            AND (expiration IS NULL OR expiration > NOW())
            """,
            (key,),
        )
        result = self.cursor.fetchone()
        return result[0] if result else None

    async def delete(self, key: str) -> None:
        key = self._namespaced_key(key)
        self.cursor.execute(
            f"DELETE FROM {self.config.table_name} WHERE key = %s",
            (key,),
        )

    async def values_in_range(self, start_key: str, end_key: str) -> list[str]:
        start_key = self._namespaced_key(start_key)
        end_key = self._namespaced_key(end_key)

        self.cursor.execute(
            f"""
            SELECT value FROM {self.config.table_name}
            WHERE key >= %s AND key < %s
            AND (expiration IS NULL OR expiration > NOW())
            ORDER BY key
            """,
            (start_key, end_key),
        )
        return [row[0] for row in self.cursor.fetchall()]

    async def keys_in_range(self, start_key: str, end_key: str) -> list[str]:
        start_key = self._namespaced_key(start_key)
        end_key = self._namespaced_key(end_key)

        self.cursor.execute(
            f"SELECT key FROM {self.config.table_name} WHERE key >= %s AND key < %s",
            (start_key, end_key),
        )
        return [row[0] for row in self.cursor.fetchall()]

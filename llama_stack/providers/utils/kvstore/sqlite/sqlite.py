# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from datetime import datetime

import aiosqlite

from ..api import KVStore
from ..config import SqliteKVStoreConfig


class SqliteKVStoreImpl(KVStore):
    def __init__(self, config: SqliteKVStoreConfig):
        self.db_path = config.db_path
        self.table_name = "kvstore"

    def __str__(self):
        return f"SqliteKVStoreImpl(db_path={self.db_path}, table_name={self.table_name})"

    async def initialize(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    expiration TIMESTAMP
                )
            """
            )
            await db.commit()

    async def set(self, key: str, value: str, expiration: datetime | None = None) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                f"INSERT OR REPLACE INTO {self.table_name} (key, value, expiration) VALUES (?, ?, ?)",
                (key, value, expiration),
            )
            await db.commit()

    async def get(self, key: str) -> str | None:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(f"SELECT value, expiration FROM {self.table_name} WHERE key = ?", (key,)) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None
                value, expiration = row
                return value

    async def delete(self, key: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(f"DELETE FROM {self.table_name} WHERE key = ?", (key,))
            await db.commit()

    async def values_in_range(self, start_key: str, end_key: str) -> list[str]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                f"SELECT key, value, expiration FROM {self.table_name} WHERE key >= ? AND key <= ?",
                (start_key, end_key),
            ) as cursor:
                result = []
                async for row in cursor:
                    _, value, _ = row
                    result.append(value)
                return result

    async def keys_in_range(self, start_key: str, end_key: str) -> list[str]:
        """Get all keys in the given range."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                f"SELECT key FROM {self.table_name} WHERE key >= ? AND key <= ?",
                (start_key, end_key),
            )
            rows = await cursor.fetchall()
            return [row[0] for row in rows]

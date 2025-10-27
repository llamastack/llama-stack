# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime

from redis.asyncio import Redis

from ..api import KVStore
from ..config import RedisKVStoreConfig


class RedisKVStoreImpl(KVStore):
    def __init__(self, config: RedisKVStoreConfig):
        self.config = config

    async def initialize(self) -> None:
        self.redis = Redis.from_url(self.config.url)

    def _namespaced_key(self, key: str) -> str:
        if not self.config.namespace:
            return key
        return f"{self.config.namespace}:{key}"

    async def set(self, key: str, value: str, expiration: datetime | None = None) -> None:
        key = self._namespaced_key(key)
        await self.redis.set(key, value)
        if expiration:
            await self.redis.expireat(key, expiration)

    async def get(self, key: str) -> str | None:
        key = self._namespaced_key(key)
        value = await self.redis.get(key)
        if value is None:
            return None
        await self.redis.ttl(key)
        return value

    async def delete(self, key: str) -> None:
        key = self._namespaced_key(key)
        await self.redis.delete(key)

    async def values_in_range(self, start_key: str, end_key: str) -> list[str]:
        start_key = self._namespaced_key(start_key)
        end_key = self._namespaced_key(end_key)
        cursor = 0
        pattern = start_key + "*"  # Match all keys starting with start_key prefix
        matching_keys = []
        while True:
            cursor, keys = await self.redis.scan(cursor, match=pattern, count=1000)

            for key in keys:
                key_str = key.decode("utf-8") if isinstance(key, bytes) else key
                if start_key <= key_str <= end_key:
                    matching_keys.append(key)

            if cursor == 0:
                break

        # Then fetch all values in a single MGET call
        if matching_keys:
            values = await self.redis.mget(matching_keys)
            return [
                value.decode("utf-8") if isinstance(value, bytes) else value for value in values if value is not None
            ]

        return []

    async def keys_in_range(self, start_key: str, end_key: str) -> list[str]:
        """Get all keys in the given range."""
        matching_keys = await self.redis.zrangebylex(self.namespace, f"[{start_key}", f"[{end_key}")
        if not matching_keys:
            return []
        return [k.decode("utf-8") for k in matching_keys]

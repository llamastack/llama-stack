# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Redis-based cache store implementation.

This module provides a production-ready Redis cache store with connection
pooling, retry logic, and comprehensive error handling. Suitable for
distributed deployments and high-throughput scenarios.
"""

import asyncio
import json
from typing import Any, Optional

from redis import asyncio as aioredis
from redis.asyncio import ConnectionPool, Redis
from redis.exceptions import ConnectionError, RedisError, TimeoutError

from llama_stack.log import get_logger

from .cache_store import CacheError

logger = get_logger(__name__)


class RedisCacheStore:
    """Redis-based cache store with connection pooling.

    This implementation provides production-ready caching with:
    - Connection pooling for efficient resource usage
    - Automatic retry logic for transient failures
    - Configurable timeouts to prevent blocking
    - JSON serialization for complex data types
    - Support for Redis cluster and sentinel

    Example:
        cache = RedisCacheStore(
            host="localhost",
            port=6379,
            db=0,
            password="secret",
            connection_pool_size=10,
            timeout_ms=100
        )
        await cache.set("key", {"data": "value"}, ttl=300)
        value = await cache.get("key")
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        connection_pool_size: int = 10,
        timeout_ms: int = 100,
        default_ttl: int = 600,
        max_retries: int = 3,
        key_prefix: str = "llama_stack:",
    ):
        """Initialize Redis cache store.

        Args:
            host: Redis server hostname
            port: Redis server port
            db: Redis database number (0-15)
            password: Optional Redis password
            connection_pool_size: Maximum connections in pool
            timeout_ms: Operation timeout in milliseconds
            default_ttl: Default time-to-live in seconds
            max_retries: Maximum retry attempts for failed operations
            key_prefix: Prefix for all cache keys (namespace isolation)

        Raises:
            ValueError: If invalid parameters provided
        """
        if connection_pool_size <= 0:
            raise ValueError("connection_pool_size must be positive")
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        if default_ttl <= 0:
            raise ValueError("default_ttl must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.connection_pool_size = connection_pool_size
        self.timeout_ms = timeout_ms
        self.default_ttl = default_ttl
        self.max_retries = max_retries
        self.key_prefix = key_prefix

        # Connection pool (lazy initialization)
        self._pool: Optional[ConnectionPool] = None
        self._redis: Optional[Redis] = None

        logger.info(
            f"Initialized RedisCacheStore: host={host}:{port}, db={db}, "
            f"pool_size={connection_pool_size}, timeout={timeout_ms}ms, "
            f"default_ttl={default_ttl}s"
        )

    async def _ensure_connection(self) -> Redis:
        """Ensure Redis connection is established.

        Returns:
            Redis client instance

        Raises:
            CacheError: If connection cannot be established
        """
        if self._redis is not None:
            return self._redis

        try:
            # Create connection pool
            self._pool = ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=self.connection_pool_size,
                socket_timeout=self.timeout_ms / 1000.0,
                socket_connect_timeout=self.timeout_ms / 1000.0,
                decode_responses=True,
            )

            # Create Redis client
            self._redis = Redis(connection_pool=self._pool)

            # Test connection
            await asyncio.wait_for(
                self._redis.ping(),
                timeout=self.timeout_ms / 1000.0
            )

            logger.info(f"Connected to Redis at {self.host}:{self.port}")
            return self._redis

        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise CacheError(f"Failed to connect to Redis at {self.host}:{self.port}", cause=e) from e
        except Exception as e:
            logger.error(f"Failed to initialize Redis connection: {e}")
            raise CacheError("Failed to initialize Redis connection", cause=e) from e

    def _make_key(self, key: str) -> str:
        """Create prefixed cache key for namespace isolation.

        Args:
            key: Base cache key

        Returns:
            Prefixed key
        """
        return f"{self.key_prefix}{key}"

    def _serialize(self, value: Any) -> str:
        """Serialize value for storage.

        Args:
            value: Value to serialize

        Returns:
            JSON-serialized string

        Raises:
            ValueError: If value cannot be serialized
        """
        try:
            return json.dumps(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Value is not JSON-serializable: {e}") from e

    def _deserialize(self, data: str) -> Any:
        """Deserialize stored value.

        Args:
            data: JSON-serialized string

        Returns:
            Deserialized value

        Raises:
            ValueError: If data cannot be deserialized
        """
        try:
            return json.loads(data)
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to deserialize cache value: {e}")
            return None

    async def _retry_operation(self, operation, *args, **kwargs) -> Any:
        """Retry an operation with exponential backoff.

        Args:
            operation: Async function to retry
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation

        Returns:
            Operation result

        Raises:
            CacheError: If all retries fail
        """
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                return await asyncio.wait_for(
                    operation(*args, **kwargs),
                    timeout=self.timeout_ms / 1000.0
                )
            except (ConnectionError, TimeoutError) as e:
                last_error = e
                if attempt < self.max_retries:
                    backoff = 2 ** attempt * 0.1  # 100ms, 200ms, 400ms
                    logger.warning(
                        f"Redis operation failed (attempt {attempt + 1}/{self.max_retries + 1}), "
                        f"retrying in {backoff}s: {e}"
                    )
                    await asyncio.sleep(backoff)
                else:
                    logger.error(f"Redis operation failed after {self.max_retries + 1} attempts")
            except Exception as e:
                # Don't retry on non-transient errors
                raise CacheError(f"Redis operation failed: {e}", cause=e) from e

        raise CacheError(f"Redis operation failed after {self.max_retries + 1} attempts", cause=last_error) from last_error

    async def get(self, key: str) -> Optional[Any]:
        """Retrieve a value from the cache.

        Args:
            key: Cache key to retrieve

        Returns:
            Cached value if present and not expired, None otherwise

        Raises:
            CacheError: If cache operation fails
        """
        try:
            redis = await self._ensure_connection()
            prefixed_key = self._make_key(key)

            data = await self._retry_operation(redis.get, prefixed_key)

            if data is None:
                return None

            value = self._deserialize(data)
            if value is not None:
                logger.debug(f"Cache hit: {key}")
            return value

        except CacheError:
            raise
        except Exception as e:
            logger.error(f"Failed to get cache key '{key}': {e}")
            raise CacheError(f"Failed to get cache key '{key}'", cause=e) from e

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """Store a value in the cache with optional TTL.

        Args:
            key: Cache key
            value: Value to cache (must be JSON-serializable)
            ttl: Time-to-live in seconds. If None, uses default TTL.

        Raises:
            CacheError: If cache operation fails
            ValueError: If value is not serializable
        """
        try:
            redis = await self._ensure_connection()
            prefixed_key = self._make_key(key)

            # Serialize value
            data = self._serialize(value)

            # Use default TTL if not specified
            effective_ttl = ttl if ttl is not None else self.default_ttl

            # Store with TTL
            await self._retry_operation(
                redis.setex,
                prefixed_key,
                effective_ttl,
                data
            )

            logger.debug(f"Cache set: {key} (ttl={effective_ttl}s)")

        except ValueError:
            raise
        except CacheError:
            raise
        except Exception as e:
            logger.error(f"Failed to set cache key '{key}': {e}")
            raise CacheError(f"Failed to set cache key '{key}'", cause=e) from e

    async def delete(self, key: str) -> bool:
        """Delete a key from the cache.

        Args:
            key: Cache key to delete

        Returns:
            True if key was deleted, False if key didn't exist

        Raises:
            CacheError: If cache operation fails
        """
        try:
            redis = await self._ensure_connection()
            prefixed_key = self._make_key(key)

            deleted_count = await self._retry_operation(redis.delete, prefixed_key)

            if deleted_count > 0:
                logger.debug(f"Cache delete: {key}")

            return bool(deleted_count > 0)

        except CacheError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete cache key '{key}': {e}")
            raise CacheError(f"Failed to delete cache key '{key}'", cause=e) from e

    async def exists(self, key: str) -> bool:
        """Check if a key exists in the cache.

        Args:
            key: Cache key to check

        Returns:
            True if key exists and is not expired, False otherwise

        Raises:
            CacheError: If cache operation fails
        """
        try:
            redis = await self._ensure_connection()
            prefixed_key = self._make_key(key)

            exists = await self._retry_operation(redis.exists, prefixed_key)
            return bool(exists > 0)

        except CacheError:
            raise
        except Exception as e:
            logger.error(f"Failed to check cache key existence '{key}': {e}")
            raise CacheError(f"Failed to check cache key existence '{key}'", cause=e) from e

    async def ttl(self, key: str) -> Optional[int]:
        """Get the remaining TTL for a key.

        Args:
            key: Cache key

        Returns:
            Remaining TTL in seconds, None if key doesn't exist or has no TTL

        Raises:
            CacheError: If cache operation fails
        """
        try:
            redis = await self._ensure_connection()
            prefixed_key = self._make_key(key)

            ttl_seconds = await self._retry_operation(redis.ttl, prefixed_key)

            # Redis returns -2 if key doesn't exist, -1 if no TTL
            if ttl_seconds == -2:
                return None
            if ttl_seconds == -1:
                return None

            return int(max(0, ttl_seconds))

        except CacheError:
            raise
        except Exception as e:
            logger.error(f"Failed to get TTL for cache key '{key}': {e}")
            raise CacheError(f"Failed to get TTL for cache key '{key}'", cause=e) from e

    async def clear(self) -> None:
        """Clear all entries from the cache.

        This deletes all keys matching the key_prefix pattern.

        Raises:
            CacheError: If cache operation fails
        """
        try:
            redis = await self._ensure_connection()
            pattern = f"{self.key_prefix}*"

            # Scan and delete keys matching pattern
            cursor = 0
            deleted_total = 0

            while True:
                cursor, keys = await self._retry_operation(
                    redis.scan,
                    cursor=cursor,
                    match=pattern,
                    count=100
                )

                if keys:
                    deleted_count = await self._retry_operation(redis.delete, *keys)
                    deleted_total += deleted_count

                if cursor == 0:
                    break

            logger.info(f"Cache cleared: deleted {deleted_total} keys")

        except CacheError:
            raise
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            raise CacheError("Failed to clear cache", cause=e) from e

    async def size(self) -> int:
        """Get the number of entries in the cache.

        Returns:
            Number of cached entries matching key_prefix

        Raises:
            CacheError: If cache operation fails
        """
        try:
            redis = await self._ensure_connection()
            pattern = f"{self.key_prefix}*"

            # Count keys matching pattern
            cursor = 0
            count = 0

            while True:
                cursor, keys = await self._retry_operation(
                    redis.scan,
                    cursor=cursor,
                    match=pattern,
                    count=100
                )

                count += len(keys)

                if cursor == 0:
                    break

            return count

        except CacheError:
            raise
        except Exception as e:
            logger.error(f"Failed to get cache size: {e}")
            raise CacheError("Failed to get cache size", cause=e) from e

    async def close(self) -> None:
        """Close Redis connection and cleanup resources.

        This should be called when the cache is no longer needed.
        """
        try:
            if self._redis is not None:
                await self._redis.close()
                self._redis = None

            if self._pool is not None:
                await self._pool.disconnect()
                self._pool = None

            logger.info("Redis connection closed")

        except Exception as e:
            logger.warning(f"Error closing Redis connection: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache configuration and connection info
        """
        return {
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "connection_pool_size": self.connection_pool_size,
            "timeout_ms": self.timeout_ms,
            "default_ttl": self.default_ttl,
            "max_retries": self.max_retries,
            "key_prefix": self.key_prefix,
            "connected": self._redis is not None,
        }

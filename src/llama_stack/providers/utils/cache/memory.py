# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""In-memory cache store implementation using cachetools.

This module provides a memory-based cache store suitable for development
and single-node deployments. For production multi-node deployments,
consider using RedisCacheStore instead.
"""

import sys
import time
from typing import Any, Literal, Optional

from cachetools import Cache, LFUCache, LRUCache, TTLCache  # type: ignore # no types-cachetools available

from llama_stack.log import get_logger

from .cache_store import CacheError

logger = get_logger(__name__)


EvictionPolicy = Literal["lru", "lfu", "ttl-only"]


class MemoryCacheStore:
    """In-memory cache store with configurable eviction policies.

    This implementation uses the cachetools library to provide efficient
    in-memory caching with support for multiple eviction policies:
    - LRU (Least Recently Used): Evicts least recently accessed items
    - LFU (Least Frequently Used): Evicts least frequently accessed items
    - TTL-only: Evicts based on time-to-live only

    Thread-safe for concurrent access within a single process.

    Example:
        cache = MemoryCacheStore(
            max_entries=1000,
            default_ttl=600,
            eviction_policy="lru"
        )
        await cache.set("key", "value", ttl=300)
        value = await cache.get("key")
    """

    def __init__(
        self,
        max_entries: int = 1000,
        max_memory_mb: Optional[int] = 512,
        default_ttl: int = 600,
        eviction_policy: EvictionPolicy = "lru",
    ):
        """Initialize memory cache store.

        Args:
            max_entries: Maximum number of entries to store
            max_memory_mb: Maximum memory usage in MB (soft limit, estimated)
            default_ttl: Default time-to-live in seconds
            eviction_policy: Eviction strategy ("lru", "lfu", "ttl-only")

        Raises:
            ValueError: If invalid parameters provided
        """
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if default_ttl <= 0:
            raise ValueError("default_ttl must be positive")
        if max_memory_mb is not None and max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")

        self.max_entries = max_entries
        self.max_memory_mb = max_memory_mb
        self.default_ttl = default_ttl
        self.eviction_policy = eviction_policy

        # Create appropriate cache implementation
        self._cache: Cache = self._create_cache()
        self._ttl_map: dict[str, float] = {}  # Track expiration times

        logger.info(
            f"Initialized MemoryCacheStore: policy={eviction_policy}, "
            f"max_entries={max_entries}, max_memory={max_memory_mb}MB, "
            f"default_ttl={default_ttl}s"
        )

    def _create_cache(self) -> Cache:
        """Create cache instance based on eviction policy.

        Returns:
            Cache instance configured with chosen policy
        """
        if self.eviction_policy == "lru":
            return LRUCache(maxsize=self.max_entries)
        elif self.eviction_policy == "lfu":
            return LFUCache(maxsize=self.max_entries)
        elif self.eviction_policy == "ttl-only":
            return TTLCache(maxsize=self.max_entries, ttl=self.default_ttl)
        else:
            raise ValueError(f"Unknown eviction policy: {self.eviction_policy}")

    def _is_expired(self, key: str) -> bool:
        """Check if a key has expired based on TTL.

        Args:
            key: Cache key to check

        Returns:
            True if key has expired, False otherwise
        """
        if key not in self._ttl_map:
            return False

        expiration_time = self._ttl_map[key]
        if time.time() >= expiration_time:
            # Clean up expired entry
            self._cache.pop(key, None)
            self._ttl_map.pop(key, None)
            return True

        return False

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
            # Check expiration first
            if self._is_expired(key):
                return None

            value = self._cache.get(key)
            if value is not None:
                logger.debug(f"Cache hit: {key}")
            return value

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
            value: Value to cache
            ttl: Time-to-live in seconds. If None, uses default TTL.

        Raises:
            CacheError: If cache operation fails
        """
        try:
            # Use default TTL if not specified
            effective_ttl = ttl if ttl is not None else self.default_ttl

            # Store value
            self._cache[key] = value

            # Track expiration time
            self._ttl_map[key] = time.time() + effective_ttl

            # Check memory usage (soft limit)
            if self.max_memory_mb is not None:
                self._check_memory_usage()

            logger.debug(f"Cache set: {key} (ttl={effective_ttl}s)")

        except Exception as e:
            logger.error(f"Failed to set cache key '{key}': {e}")
            raise CacheError(f"Failed to set cache key '{key}'", cause=e) from e

    def _check_memory_usage(self) -> None:
        """Check and log if memory usage exceeds soft limit.

        This is a soft limit - we log warnings but don't enforce hard limits.
        The cachetools library will handle eviction based on max_entries.
        """
        try:
            # Get approximate memory usage
            cache_size_bytes = sys.getsizeof(self._cache) + sys.getsizeof(self._ttl_map)

            # Convert to MB
            cache_size_mb = cache_size_bytes / (1024 * 1024)

            if self.max_memory_mb is not None and cache_size_mb > self.max_memory_mb:
                logger.warning(
                    f"Cache memory usage ({cache_size_mb:.1f}MB) exceeds "
                    f"soft limit ({self.max_memory_mb}MB). "
                    f"Consider increasing max_entries or max_memory_mb."
                )
        except Exception as e:
            # Don't fail on memory check errors
            logger.debug(f"Memory usage check failed: {e}")

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
            existed = key in self._cache
            self._cache.pop(key, None)
            self._ttl_map.pop(key, None)

            if existed:
                logger.debug(f"Cache delete: {key}")

            return existed

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
            if self._is_expired(key):
                return False
            return key in self._cache

        except Exception as e:
            logger.error(f"Failed to check cache key existence '{key}': {e}")
            raise CacheError(f"Failed to check cache key existence '{key}'", cause=e) from e

    async def ttl(self, key: str) -> Optional[int]:
        """Get the remaining TTL for a key.

        Args:
            key: Cache key

        Returns:
            Remaining TTL in seconds, None if key doesn't exist

        Raises:
            CacheError: If cache operation fails
        """
        try:
            if key not in self._ttl_map:
                return None

            if self._is_expired(key):
                return None

            remaining = int(self._ttl_map[key] - time.time())
            return max(0, remaining)

        except Exception as e:
            logger.error(f"Failed to get TTL for cache key '{key}': {e}")
            raise CacheError(f"Failed to get TTL for cache key '{key}'", cause=e) from e

    async def clear(self) -> None:
        """Clear all entries from the cache.

        Raises:
            CacheError: If cache operation fails
        """
        try:
            self._cache.clear()
            self._ttl_map.clear()
            logger.info("Cache cleared")

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            raise CacheError("Failed to clear cache", cause=e) from e

    async def size(self) -> int:
        """Get the number of entries in the cache.

        Returns:
            Number of cached entries (excluding expired entries)

        Raises:
            CacheError: If cache operation fails
        """
        try:
            # Clean up expired entries first
            expired_keys = [
                key for key in list(self._ttl_map.keys())
                if self._is_expired(key)
            ]

            return len(self._cache)

        except Exception as e:
            logger.error(f"Failed to get cache size: {e}")
            raise CacheError("Failed to get cache size", cause=e) from e

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics including size, policy, and limits
        """
        return {
            "size": len(self._cache),
            "max_entries": self.max_entries,
            "max_memory_mb": self.max_memory_mb,
            "default_ttl": self.default_ttl,
            "eviction_policy": self.eviction_policy,
        }

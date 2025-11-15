# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Cache store utilities for prompt caching.

This module provides cache store abstractions and implementations for use in
the Llama Stack server's prompt caching feature. Supports both memory-based
and Redis-based caching with configurable eviction policies and TTL management.

Example usage:
    from llama_stack.providers.utils.cache import MemoryCacheStore, RedisCacheStore

    # Memory cache for development
    memory_cache = MemoryCacheStore(max_entries=1000, eviction_policy="lru")

    # Redis cache for production
    redis_cache = RedisCacheStore(
        host="localhost",
        port=6379,
        connection_pool_size=10
    )
"""

from .cache_store import CacheError, CacheStore, CircuitBreaker
from .memory import MemoryCacheStore
from .redis import RedisCacheStore

__all__ = [
    "CacheStore",
    "CacheError",
    "CircuitBreaker",
    "MemoryCacheStore",
    "RedisCacheStore",
]

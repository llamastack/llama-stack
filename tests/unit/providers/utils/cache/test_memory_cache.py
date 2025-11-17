# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for MemoryCacheStore implementation."""

import asyncio

import pytest

from llama_stack.providers.utils.cache import CacheError, MemoryCacheStore


class TestMemoryCacheStore:
    """Test suite for MemoryCacheStore."""

    async def test_init_default_params(self):
        """Test initialization with default parameters."""
        cache = MemoryCacheStore()
        assert cache.max_entries == 1000
        assert cache.max_memory_mb == 512
        assert cache.default_ttl == 600
        assert cache.eviction_policy == "lru"

    async def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        cache = MemoryCacheStore(
            max_entries=500,
            max_memory_mb=256,
            default_ttl=300,
            eviction_policy="lfu",
        )
        assert cache.max_entries == 500
        assert cache.max_memory_mb == 256
        assert cache.default_ttl == 300
        assert cache.eviction_policy == "lfu"

    async def test_init_invalid_params(self):
        """Test initialization with invalid parameters."""
        with pytest.raises(ValueError, match="max_entries must be positive"):
            MemoryCacheStore(max_entries=0)

        with pytest.raises(ValueError, match="default_ttl must be positive"):
            MemoryCacheStore(default_ttl=0)

        with pytest.raises(ValueError, match="max_memory_mb must be positive"):
            MemoryCacheStore(max_memory_mb=0)

        with pytest.raises(ValueError, match="Unknown eviction policy"):
            MemoryCacheStore(eviction_policy="invalid")  # type: ignore

    async def test_set_and_get(self):
        """Test basic set and get operations."""
        cache = MemoryCacheStore()

        # Set value
        await cache.set("key1", "value1")

        # Get value
        value = await cache.get("key1")
        assert value == "value1"

    async def test_get_nonexistent_key(self):
        """Test getting a non-existent key."""
        cache = MemoryCacheStore()
        value = await cache.get("nonexistent")
        assert value is None

    async def test_set_with_custom_ttl(self):
        """Test setting value with custom TTL."""
        cache = MemoryCacheStore(default_ttl=10)

        # Set with custom TTL
        await cache.set("key1", "value1", ttl=1)

        # Value should exist initially
        value = await cache.get("key1")
        assert value == "value1"

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Value should be expired
        value = await cache.get("key1")
        assert value is None

    async def test_set_complex_value(self):
        """Test storing complex data types."""
        cache = MemoryCacheStore()

        # Test dictionary
        data = {"nested": {"key": "value"}, "list": [1, 2, 3]}
        await cache.set("complex", data)
        value = await cache.get("complex")
        assert value == data

        # Test list
        list_data = [1, "two", {"three": 3}]
        await cache.set("list", list_data)
        value = await cache.get("list")
        assert value == list_data

    async def test_delete(self):
        """Test deleting a key."""
        cache = MemoryCacheStore()

        # Set and delete
        await cache.set("key1", "value1")
        deleted = await cache.delete("key1")
        assert deleted is True

        # Verify deleted
        value = await cache.get("key1")
        assert value is None

        # Delete non-existent key
        deleted = await cache.delete("nonexistent")
        assert deleted is False

    async def test_exists(self):
        """Test checking key existence."""
        cache = MemoryCacheStore()

        # Non-existent key
        exists = await cache.exists("key1")
        assert exists is False

        # Existing key
        await cache.set("key1", "value1")
        exists = await cache.exists("key1")
        assert exists is True

        # Expired key
        await cache.set("key2", "value2", ttl=1)
        await asyncio.sleep(1.1)
        exists = await cache.exists("key2")
        assert exists is False

    async def test_ttl(self):
        """Test getting remaining TTL."""
        cache = MemoryCacheStore()

        # Non-existent key
        ttl = await cache.ttl("nonexistent")
        assert ttl is None

        # Key with TTL
        await cache.set("key1", "value1", ttl=10)
        ttl = await cache.ttl("key1")
        assert ttl is not None
        assert 8 <= ttl <= 10  # Allow some tolerance

        # Expired key
        await cache.set("key2", "value2", ttl=1)
        await asyncio.sleep(1.1)
        ttl = await cache.ttl("key2")
        assert ttl is None

    async def test_clear(self):
        """Test clearing all entries."""
        cache = MemoryCacheStore()

        # Add multiple entries
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        # Clear
        await cache.clear()

        # Verify all cleared
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
        assert await cache.get("key3") is None

    async def test_size(self):
        """Test getting cache size."""
        cache = MemoryCacheStore()

        # Empty cache
        size = await cache.size()
        assert size == 0

        # Add entries
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        size = await cache.size()
        assert size == 2

        # Delete entry
        await cache.delete("key1")
        size = await cache.size()
        assert size == 1

        # Clear cache
        await cache.clear()
        size = await cache.size()
        assert size == 0

    async def test_lru_eviction(self):
        """Test LRU eviction policy."""
        cache = MemoryCacheStore(max_entries=3, eviction_policy="lru")

        # Fill cache
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        # Access key1 to make it recently used
        await cache.get("key1")

        # Add new entry, should evict key2 (least recently used)
        await cache.set("key4", "value4")

        # key2 should be evicted
        assert await cache.get("key1") == "value1"
        assert await cache.get("key2") is None
        assert await cache.get("key3") == "value3"
        assert await cache.get("key4") == "value4"

    async def test_lfu_eviction(self):
        """Test LFU eviction policy."""
        cache = MemoryCacheStore(max_entries=3, eviction_policy="lfu")

        # Fill cache
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        # Access key1 multiple times
        await cache.get("key1")
        await cache.get("key1")
        await cache.get("key1")

        # Access key2 twice
        await cache.get("key2")
        await cache.get("key2")

        # key3 accessed once (least frequently)

        # Add new entry, should evict key3 (least frequently used)
        await cache.set("key4", "value4")

        # key3 should be evicted
        assert await cache.get("key1") == "value1"
        assert await cache.get("key2") == "value2"
        assert await cache.get("key3") is None
        assert await cache.get("key4") == "value4"

    async def test_concurrent_access(self):
        """Test concurrent access to cache."""
        cache = MemoryCacheStore()

        async def set_value(key: str, value: str):
            await cache.set(key, value)

        async def get_value(key: str):
            return await cache.get(key)

        # Concurrent sets
        await asyncio.gather(
            set_value("key1", "value1"),
            set_value("key2", "value2"),
            set_value("key3", "value3"),
        )

        # Concurrent gets
        results = await asyncio.gather(
            get_value("key1"),
            get_value("key2"),
            get_value("key3"),
        )

        assert results == ["value1", "value2", "value3"]

    async def test_update_existing_key(self):
        """Test updating an existing key."""
        cache = MemoryCacheStore()

        # Set initial value
        await cache.set("key1", "value1")
        assert await cache.get("key1") == "value1"

        # Update value
        await cache.set("key1", "value2")
        assert await cache.get("key1") == "value2"

    async def test_get_stats(self):
        """Test getting cache statistics."""
        cache = MemoryCacheStore(
            max_entries=100,
            max_memory_mb=128,
            default_ttl=300,
            eviction_policy="lru",
        )

        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        stats = cache.get_stats()

        assert stats["size"] == 2
        assert stats["max_entries"] == 100
        assert stats["max_memory_mb"] == 128
        assert stats["default_ttl"] == 300
        assert stats["eviction_policy"] == "lru"

    async def test_ttl_expiration_cleanup(self):
        """Test that expired entries are cleaned up properly."""
        cache = MemoryCacheStore()

        # Set entry with short TTL
        await cache.set("key1", "value1", ttl=1)
        await cache.set("key2", "value2", ttl=10)

        # Initially both exist
        assert await cache.size() == 2

        # Wait for key1 to expire
        await asyncio.sleep(1.1)

        # Accessing expired key should clean it up
        assert await cache.get("key1") is None

        # Size should reflect cleanup
        size = await cache.size()
        assert size == 1

        # key2 should still exist
        assert await cache.get("key2") == "value2"

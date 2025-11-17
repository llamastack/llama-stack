# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for RedisCacheStore implementation."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llama_stack.providers.utils.cache import CacheError, RedisCacheStore


class TestRedisCacheStore:
    """Test suite for RedisCacheStore."""

    async def test_init_default_params(self):
        """Test initialization with default parameters."""
        cache = RedisCacheStore()
        assert cache.host == "localhost"
        assert cache.port == 6379
        assert cache.db == 0
        assert cache.password is None
        assert cache.connection_pool_size == 10
        assert cache.timeout_ms == 100
        assert cache.default_ttl == 600
        assert cache.max_retries == 3
        assert cache.key_prefix == "llama_stack:"

    async def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        cache = RedisCacheStore(
            host="redis.example.com",
            port=6380,
            db=1,
            password="secret",
            connection_pool_size=20,
            timeout_ms=200,
            default_ttl=300,
            max_retries=5,
            key_prefix="test:",
        )
        assert cache.host == "redis.example.com"
        assert cache.port == 6380
        assert cache.db == 1
        assert cache.password == "secret"
        assert cache.connection_pool_size == 20
        assert cache.timeout_ms == 200
        assert cache.default_ttl == 300
        assert cache.max_retries == 5
        assert cache.key_prefix == "test:"

    async def test_init_invalid_params(self):
        """Test initialization with invalid parameters."""
        with pytest.raises(ValueError, match="connection_pool_size must be positive"):
            RedisCacheStore(connection_pool_size=0)

        with pytest.raises(ValueError, match="timeout_ms must be positive"):
            RedisCacheStore(timeout_ms=0)

        with pytest.raises(ValueError, match="default_ttl must be positive"):
            RedisCacheStore(default_ttl=0)

        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            RedisCacheStore(max_retries=-1)

    @patch("llama_stack.providers.utils.cache.redis.ConnectionPool")
    @patch("llama_stack.providers.utils.cache.redis.Redis")
    async def test_ensure_connection(self, mock_redis_class, mock_pool_class):
        """Test connection establishment."""
        # Setup mocks
        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis_class.return_value = mock_redis

        # Create cache
        cache = RedisCacheStore()

        # Ensure connection
        redis = await cache._ensure_connection()

        # Verify connection was established
        assert redis == mock_redis
        mock_pool_class.assert_called_once()
        mock_redis.ping.assert_called_once()

    @patch("llama_stack.providers.utils.cache.redis.ConnectionPool")
    @patch("llama_stack.providers.utils.cache.redis.Redis")
    async def test_connection_failure(self, mock_redis_class, mock_pool_class):
        """Test connection failure handling."""
        from redis.exceptions import ConnectionError as RedisConnectionError

        # Setup mocks to fail
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(side_effect=RedisConnectionError("Connection refused"))
        mock_redis_class.return_value = mock_redis

        # Create cache
        cache = RedisCacheStore()

        # Connection should fail
        with pytest.raises(CacheError, match="Failed to connect to Redis"):
            await cache._ensure_connection()

    def test_make_key(self):
        """Test key prefixing."""
        cache = RedisCacheStore(key_prefix="test:")
        assert cache._make_key("mykey") == "test:mykey"
        assert cache._make_key("another") == "test:another"

    def test_serialize_deserialize(self):
        """Test value serialization."""
        cache = RedisCacheStore()

        # Simple value
        assert cache._serialize("hello") == '"hello"'
        assert cache._deserialize('"hello"') == "hello"

        # Dictionary
        data = {"key": "value", "number": 42}
        serialized = cache._serialize(data)
        assert cache._deserialize(serialized) == data

        # List
        list_data = [1, 2, "three"]
        serialized = cache._serialize(list_data)
        assert cache._deserialize(serialized) == list_data

    def test_serialize_error(self):
        """Test serialization error handling."""
        cache = RedisCacheStore()

        # Object that can't be serialized
        class NonSerializable:
            pass

        with pytest.raises(ValueError, match="Value is not JSON-serializable"):
            cache._serialize(NonSerializable())

    @patch("llama_stack.providers.utils.cache.redis.ConnectionPool")
    @patch("llama_stack.providers.utils.cache.redis.Redis")
    async def test_set_and_get(self, mock_redis_class, mock_pool_class):
        """Test set and get operations."""
        # Setup mocks
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis.get = AsyncMock(return_value=json.dumps("value1"))
        mock_redis.setex = AsyncMock()
        mock_redis_class.return_value = mock_redis

        # Create cache
        cache = RedisCacheStore()

        # Set value
        await cache.set("key1", "value1")
        mock_redis.setex.assert_called_once()

        # Get value
        value = await cache.get("key1")
        assert value == "value1"
        mock_redis.get.assert_called_once()

    @patch("llama_stack.providers.utils.cache.redis.ConnectionPool")
    @patch("llama_stack.providers.utils.cache.redis.Redis")
    async def test_get_nonexistent_key(self, mock_redis_class, mock_pool_class):
        """Test getting a non-existent key."""
        # Setup mocks
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis_class.return_value = mock_redis

        # Create cache
        cache = RedisCacheStore()

        # Get non-existent key
        value = await cache.get("nonexistent")
        assert value is None

    @patch("llama_stack.providers.utils.cache.redis.ConnectionPool")
    @patch("llama_stack.providers.utils.cache.redis.Redis")
    async def test_set_with_custom_ttl(self, mock_redis_class, mock_pool_class):
        """Test setting value with custom TTL."""
        # Setup mocks
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis.setex = AsyncMock()
        mock_redis_class.return_value = mock_redis

        # Create cache
        cache = RedisCacheStore(default_ttl=600)

        # Set with custom TTL
        await cache.set("key1", "value1", ttl=300)

        # Verify setex was called with custom TTL
        call_args = mock_redis.setex.call_args
        assert call_args[0][1] == 300  # TTL argument

    @patch("llama_stack.providers.utils.cache.redis.ConnectionPool")
    @patch("llama_stack.providers.utils.cache.redis.Redis")
    async def test_delete(self, mock_redis_class, mock_pool_class):
        """Test deleting a key."""
        # Setup mocks
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis.delete = AsyncMock(return_value=1)  # 1 key deleted
        mock_redis_class.return_value = mock_redis

        # Create cache
        cache = RedisCacheStore()

        # Delete key
        deleted = await cache.delete("key1")
        assert deleted is True

        # Delete non-existent key
        mock_redis.delete = AsyncMock(return_value=0)
        deleted = await cache.delete("nonexistent")
        assert deleted is False

    @patch("llama_stack.providers.utils.cache.redis.ConnectionPool")
    @patch("llama_stack.providers.utils.cache.redis.Redis")
    async def test_exists(self, mock_redis_class, mock_pool_class):
        """Test checking key existence."""
        # Setup mocks
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis.exists = AsyncMock(return_value=1)  # Exists
        mock_redis_class.return_value = mock_redis

        # Create cache
        cache = RedisCacheStore()

        # Check existing key
        exists = await cache.exists("key1")
        assert exists is True

        # Check non-existent key
        mock_redis.exists = AsyncMock(return_value=0)
        exists = await cache.exists("nonexistent")
        assert exists is False

    @patch("llama_stack.providers.utils.cache.redis.ConnectionPool")
    @patch("llama_stack.providers.utils.cache.redis.Redis")
    async def test_ttl(self, mock_redis_class, mock_pool_class):
        """Test getting remaining TTL."""
        # Setup mocks
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis.ttl = AsyncMock(return_value=300)
        mock_redis_class.return_value = mock_redis

        # Create cache
        cache = RedisCacheStore()

        # Get TTL
        ttl = await cache.ttl("key1")
        assert ttl == 300

        # Key doesn't exist
        mock_redis.ttl = AsyncMock(return_value=-2)
        ttl = await cache.ttl("nonexistent")
        assert ttl is None

        # Key has no TTL
        mock_redis.ttl = AsyncMock(return_value=-1)
        ttl = await cache.ttl("no_ttl_key")
        assert ttl is None

    @patch("llama_stack.providers.utils.cache.redis.ConnectionPool")
    @patch("llama_stack.providers.utils.cache.redis.Redis")
    async def test_clear(self, mock_redis_class, mock_pool_class):
        """Test clearing all entries."""
        # Setup mocks
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis.scan = AsyncMock(
            side_effect=[
                (10, ["llama_stack:key1", "llama_stack:key2"]),
                (0, ["llama_stack:key3"]),  # cursor 0 indicates end
            ]
        )
        mock_redis.delete = AsyncMock(return_value=3)
        mock_redis_class.return_value = mock_redis

        # Create cache
        cache = RedisCacheStore()

        # Clear cache
        await cache.clear()

        # Verify scan and delete were called
        assert mock_redis.scan.call_count == 2
        mock_redis.delete.assert_called()

    @patch("llama_stack.providers.utils.cache.redis.ConnectionPool")
    @patch("llama_stack.providers.utils.cache.redis.Redis")
    async def test_size(self, mock_redis_class, mock_pool_class):
        """Test getting cache size."""
        # Setup mocks
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis.scan = AsyncMock(
            side_effect=[
                (10, ["llama_stack:key1", "llama_stack:key2"]),
                (0, ["llama_stack:key3"]),
            ]
        )
        mock_redis_class.return_value = mock_redis

        # Create cache
        cache = RedisCacheStore()

        # Get size
        size = await cache.size()
        assert size == 3

    @patch("llama_stack.providers.utils.cache.redis.ConnectionPool")
    @patch("llama_stack.providers.utils.cache.redis.Redis")
    async def test_retry_logic(self, mock_redis_class, mock_pool_class):
        """Test retry logic for transient failures."""
        from redis.exceptions import TimeoutError as RedisTimeoutError

        # Setup mocks - fail twice, then succeed
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis.get = AsyncMock(
            side_effect=[
                RedisTimeoutError("Timeout"),
                RedisTimeoutError("Timeout"),
                json.dumps("success"),
            ]
        )
        mock_redis_class.return_value = mock_redis

        # Create cache with retries
        cache = RedisCacheStore(max_retries=3)

        # Should succeed after retries
        value = await cache.get("key1")
        assert value == "success"
        assert mock_redis.get.call_count == 3

    @patch("llama_stack.providers.utils.cache.redis.ConnectionPool")
    @patch("llama_stack.providers.utils.cache.redis.Redis")
    async def test_retry_exhaustion(self, mock_redis_class, mock_pool_class):
        """Test behavior when all retries are exhausted."""
        from redis.exceptions import TimeoutError as RedisTimeoutError

        # Setup mocks - always fail
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=RedisTimeoutError("Timeout"))
        mock_redis_class.return_value = mock_redis

        # Create cache with limited retries
        cache = RedisCacheStore(max_retries=2)

        # Should raise CacheError after exhausting retries
        with pytest.raises(CacheError, match="failed after .* attempts"):
            await cache.get("key1")

        # Should have tried 3 times (initial + 2 retries)
        assert mock_redis.get.call_count == 3

    @patch("llama_stack.providers.utils.cache.redis.ConnectionPool")
    @patch("llama_stack.providers.utils.cache.redis.Redis")
    async def test_close(self, mock_redis_class, mock_pool_class):
        """Test closing Redis connection."""
        # Setup mocks
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis.close = AsyncMock()
        mock_redis_class.return_value = mock_redis

        mock_pool = AsyncMock()
        mock_pool.disconnect = AsyncMock()
        mock_pool_class.return_value = mock_pool

        # Create cache and establish connection
        cache = RedisCacheStore()
        await cache._ensure_connection()

        # Close connection
        await cache.close()

        # Verify cleanup
        mock_redis.close.assert_called_once()
        mock_pool.disconnect.assert_called_once()

    def test_get_stats(self):
        """Test getting cache statistics."""
        cache = RedisCacheStore(
            host="redis.example.com",
            port=6380,
            db=1,
            connection_pool_size=20,
            timeout_ms=200,
            default_ttl=300,
            max_retries=5,
            key_prefix="test:",
        )

        stats = cache.get_stats()

        assert stats["host"] == "redis.example.com"
        assert stats["port"] == 6380
        assert stats["db"] == 1
        assert stats["connection_pool_size"] == 20
        assert stats["timeout_ms"] == 200
        assert stats["default_ttl"] == 300
        assert stats["max_retries"] == 5
        assert stats["key_prefix"] == "test:"
        assert stats["connected"] is False  # Not connected yet

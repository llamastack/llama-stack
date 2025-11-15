# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Cache store abstraction for prompt caching implementation.

This module provides a protocol-based abstraction for cache storage backends,
enabling flexible storage implementations (memory, Redis, etc.) for prompt
caching in the Llama Stack server.
"""

from datetime import timedelta
from typing import Any, Optional, Protocol

from llama_stack.log import get_logger

logger = get_logger(__name__)


class CacheStore(Protocol):
    """Protocol defining the cache store interface.

    This protocol specifies the required methods for cache store implementations.
    All implementations must support TTL-based expiration and provide efficient
    key-value storage operations.

    Methods support both synchronous and asynchronous usage patterns depending
    on the implementation requirements.
    """

    async def get(self, key: str) -> Optional[Any]:
        """Retrieve a value from the cache.

        Args:
            key: Cache key to retrieve

        Returns:
            Cached value if present and not expired, None otherwise

        Raises:
            CacheError: If cache backend is unavailable or operation fails
        """
        ...

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """Store a value in the cache with optional TTL.

        Args:
            key: Cache key
            value: Value to cache (must be serializable)
            ttl: Time-to-live in seconds. If None, uses default TTL.

        Raises:
            CacheError: If cache backend is unavailable or operation fails
            ValueError: If value is not serializable
        """
        ...

    async def delete(self, key: str) -> bool:
        """Delete a key from the cache.

        Args:
            key: Cache key to delete

        Returns:
            True if key was deleted, False if key didn't exist

        Raises:
            CacheError: If cache backend is unavailable or operation fails
        """
        ...

    async def exists(self, key: str) -> bool:
        """Check if a key exists in the cache.

        Args:
            key: Cache key to check

        Returns:
            True if key exists and is not expired, False otherwise

        Raises:
            CacheError: If cache backend is unavailable or operation fails
        """
        ...

    async def ttl(self, key: str) -> Optional[int]:
        """Get the remaining TTL for a key.

        Args:
            key: Cache key

        Returns:
            Remaining TTL in seconds, None if key doesn't exist or has no TTL

        Raises:
            CacheError: If cache backend is unavailable or operation fails
        """
        ...

    async def clear(self) -> None:
        """Clear all entries from the cache.

        This is primarily useful for testing. Use with caution in production
        as it affects all cached data.

        Raises:
            CacheError: If cache backend is unavailable or operation fails
        """
        ...

    async def size(self) -> int:
        """Get the number of entries in the cache.

        Returns:
            Number of cached entries

        Raises:
            CacheError: If cache backend is unavailable or operation fails
        """
        ...


class CacheError(Exception):
    """Exception raised for cache operation failures.

    This exception is raised when cache operations fail due to backend
    unavailability, network issues, or other operational problems.
    The system should gracefully degrade when catching this exception.
    """

    def __init__(self, message: str, cause: Optional[Exception] = None):
        """Initialize cache error.

        Args:
            message: Error description (should start with "Failed to ...")
            cause: Optional underlying exception that caused this error
        """
        super().__init__(message)
        self.cause = cause


class CircuitBreaker:
    """Circuit breaker pattern for cache backend failure protection.

    Prevents cascade failures by temporarily disabling cache operations
    after detecting repeated failures. Automatically attempts recovery
    after a timeout period.

    States:
    - CLOSED: Normal operation, requests go through
    - OPEN: Too many failures, requests are blocked
    - HALF_OPEN: Testing if backend has recovered

    Example:
        breaker = CircuitBreaker(failure_threshold=10, recovery_timeout=60)
        if breaker.is_closed():
            try:
                result = await cache.get(key)
                breaker.record_success()
            except CacheError:
                breaker.record_failure()
    """

    def __init__(
        self,
        failure_threshold: int = 10,
        recovery_timeout: int = 60,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of consecutive failures before opening
            recovery_timeout: Seconds to wait before attempting recovery
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def is_closed(self) -> bool:
        """Check if circuit breaker allows operations.

        Returns:
            True if operations should proceed, False if blocked
        """
        import time

        if self.state == "CLOSED":
            return True

        if self.state == "OPEN":
            # Check if we should try recovery
            if (
                self.last_failure_time is not None
                and time.time() - self.last_failure_time >= self.recovery_timeout
            ):
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker entering HALF_OPEN state for recovery test")
                return True
            return False

        # HALF_OPEN state - allow one request through to test
        return True

    def record_success(self) -> None:
        """Record a successful operation."""
        if self.state == "HALF_OPEN":
            logger.info("Circuit breaker recovery successful, returning to CLOSED state")
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"

    def record_failure(self) -> None:
        """Record a failed operation."""
        import time

        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == "HALF_OPEN":
            # Recovery attempt failed, go back to OPEN
            logger.warning("Circuit breaker recovery failed, returning to OPEN state")
            self.state = "OPEN"
        elif self.failure_count >= self.failure_threshold:
            logger.error(
                f"Circuit breaker OPEN after {self.failure_count} failures. "
                f"Cache operations disabled for {self.recovery_timeout}s"
            )
            self.state = "OPEN"

    def get_state(self) -> str:
        """Get current circuit breaker state.

        Returns:
            Current state: "CLOSED", "OPEN", or "HALF_OPEN"
        """
        return self.state

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED state.

        This is primarily useful for testing or administrative overrides.
        """
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"
        logger.info("Circuit breaker manually reset to CLOSED state")

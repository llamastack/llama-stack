# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for cache store base classes and utilities."""

import asyncio

import pytest

from llama_stack.providers.utils.cache import CacheError, CircuitBreaker


class TestCacheError:
    """Test suite for CacheError exception."""

    def test_init_with_message(self):
        """Test CacheError initialization with message."""
        error = CacheError("Failed to connect to cache")
        assert str(error) == "Failed to connect to cache"
        assert error.cause is None

    def test_init_with_cause(self):
        """Test CacheError initialization with underlying cause."""
        cause = ValueError("Invalid value")
        error = CacheError("Failed to set cache key", cause=cause)
        assert str(error) == "Failed to set cache key"
        assert error.cause == cause


class TestCircuitBreaker:
    """Test suite for CircuitBreaker."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        breaker = CircuitBreaker()
        assert breaker.failure_threshold == 10
        assert breaker.recovery_timeout == 60
        assert breaker.failure_count == 0
        assert breaker.last_failure_time is None
        assert breaker.state == "CLOSED"

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30)
        assert breaker.failure_threshold == 5
        assert breaker.recovery_timeout == 30

    def test_is_closed_initial_state(self):
        """Test is_closed in initial state."""
        breaker = CircuitBreaker()
        assert breaker.is_closed() is True
        assert breaker.get_state() == "CLOSED"

    def test_record_success(self):
        """Test recording successful operations."""
        breaker = CircuitBreaker()

        # Record some failures
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.failure_count == 2

        # Record success should reset
        breaker.record_success()
        assert breaker.failure_count == 0
        assert breaker.last_failure_time is None
        assert breaker.state == "CLOSED"

    def test_record_failure_below_threshold(self):
        """Test recording failures below threshold."""
        breaker = CircuitBreaker(failure_threshold=5)

        # Record failures below threshold
        for i in range(4):
            breaker.record_failure()
            assert breaker.is_closed() is True
            assert breaker.state == "CLOSED"

        assert breaker.failure_count == 4

    def test_record_failure_reach_threshold(self):
        """Test circuit breaker opens when threshold reached."""
        breaker = CircuitBreaker(failure_threshold=3)

        # Record failures to reach threshold
        for i in range(3):
            breaker.record_failure()

        # Should be open now
        assert breaker.state == "OPEN"
        assert breaker.is_closed() is False

    def test_circuit_open_blocks_requests(self):
        """Test that open circuit blocks requests."""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=10)

        # Open the circuit
        for i in range(3):
            breaker.record_failure()

        assert breaker.is_closed() is False
        assert breaker.state == "OPEN"

    async def test_recovery_timeout(self):
        """Test circuit breaker recovery after timeout."""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1)

        # Open the circuit
        for i in range(3):
            breaker.record_failure()

        assert breaker.state == "OPEN"
        assert breaker.is_closed() is False

        # Wait for recovery timeout
        await asyncio.sleep(1.1)

        # Should enter HALF_OPEN state
        assert breaker.is_closed() is True
        assert breaker.state == "HALF_OPEN"

    async def test_half_open_success_closes_circuit(self):
        """Test successful request in HALF_OPEN closes circuit."""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1)

        # Open the circuit
        for i in range(3):
            breaker.record_failure()

        # Wait for recovery
        await asyncio.sleep(1.1)

        # Trigger state transition by calling is_closed()
        assert breaker.is_closed() is True
        assert breaker.state == "HALF_OPEN"

        # Record success
        breaker.record_success()
        assert breaker.state == "CLOSED"
        assert breaker.failure_count == 0

    async def test_half_open_failure_reopens_circuit(self):
        """Test failed request in HALF_OPEN reopens circuit."""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1)

        # Open the circuit
        for i in range(3):
            breaker.record_failure()

        # Wait for recovery
        await asyncio.sleep(1.1)

        # Trigger state transition by calling is_closed()
        assert breaker.is_closed() is True
        assert breaker.state == "HALF_OPEN"

        # Record failure
        breaker.record_failure()
        assert breaker.state == "OPEN"

    def test_reset(self):
        """Test manual reset of circuit breaker."""
        breaker = CircuitBreaker(failure_threshold=3)

        # Open the circuit
        for i in range(3):
            breaker.record_failure()

        assert breaker.state == "OPEN"

        # Manual reset
        breaker.reset()
        assert breaker.state == "CLOSED"
        assert breaker.failure_count == 0
        assert breaker.last_failure_time is None

    def test_get_state(self):
        """Test getting circuit breaker state."""
        breaker = CircuitBreaker(failure_threshold=3)

        # Initial state
        assert breaker.get_state() == "CLOSED"

        # After failures
        breaker.record_failure()
        assert breaker.get_state() == "CLOSED"

        # Open state
        for i in range(2):
            breaker.record_failure()
        assert breaker.get_state() == "OPEN"

    async def test_multiple_recovery_attempts(self):
        """Test multiple recovery attempts."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1)

        # Open the circuit
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == "OPEN"

        # First recovery attempt fails
        await asyncio.sleep(1.1)
        assert breaker.is_closed() is True  # Trigger state check
        assert breaker.state == "HALF_OPEN"
        breaker.record_failure()
        assert breaker.state == "OPEN"

        # Second recovery attempt succeeds
        await asyncio.sleep(1.1)
        assert breaker.is_closed() is True  # Trigger state check
        assert breaker.state == "HALF_OPEN"
        breaker.record_success()
        assert breaker.state == "CLOSED"

    def test_failure_count_tracking(self):
        """Test failure count tracking."""
        breaker = CircuitBreaker(failure_threshold=5)

        # Track failures
        assert breaker.failure_count == 0

        breaker.record_failure()
        assert breaker.failure_count == 1

        breaker.record_failure()
        assert breaker.failure_count == 2

        # Success resets count
        breaker.record_success()
        assert breaker.failure_count == 0

    async def test_concurrent_operations(self):
        """Test circuit breaker with concurrent operations."""
        breaker = CircuitBreaker(failure_threshold=10)

        async def record_failures(count: int):
            for _ in range(count):
                breaker.record_failure()
                await asyncio.sleep(0.01)

        # Concurrent failures
        await asyncio.gather(
            record_failures(3),
            record_failures(3),
            record_failures(3),
        )

        assert breaker.failure_count == 9
        assert breaker.state == "CLOSED"

        # One more should open it
        breaker.record_failure()
        assert breaker.state == "OPEN"

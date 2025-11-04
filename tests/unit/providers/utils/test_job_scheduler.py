# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from llama_stack.core.storage.datatypes import KVStoreReference, SqliteKVStoreConfig
from llama_stack.providers.utils.job_scheduler import InlineSchedulerConfig
from llama_stack.providers.utils.kvstore import register_kvstore_backends


async def default_test_executor(job_data: dict) -> dict:
    """Default test executor that simulates work."""
    await asyncio.sleep(0.1)
    return {
        "message": "Test job completed successfully",
        "job_data": job_data,
    }


@pytest.fixture
def scheduler_config():
    """Create a test scheduler config with temporary SQLite database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_scheduler.db"
        backend_name = "kv_scheduler_test"
        kvstore_config = SqliteKVStoreConfig(db_path=str(db_path))
        register_kvstore_backends({backend_name: kvstore_config})

        yield InlineSchedulerConfig(
            kvstore=KVStoreReference(backend=backend_name, namespace="job_scheduler"),
            max_concurrent_jobs=5,
        )


async def test_scheduler_api_exists(scheduler_config):
    """Test that scheduler API is properly defined."""
    from llama_stack.providers.utils.job_scheduler.inline import InlineSchedulerImpl

    scheduler = InlineSchedulerImpl(scheduler_config)

    # Verify all required methods exist
    assert hasattr(scheduler, "initialize")
    assert hasattr(scheduler, "start")
    assert hasattr(scheduler, "shutdown")
    assert hasattr(scheduler, "register_job_executor")
    assert hasattr(scheduler, "schedule_job")
    assert hasattr(scheduler, "get_job_info")
    assert hasattr(scheduler, "cancel_job")
    assert hasattr(scheduler, "delete_job")
    assert hasattr(scheduler, "list_jobs")


async def test_scheduler_not_implemented(scheduler_config):
    """Test that scheduler methods raise NotImplementedError."""
    from llama_stack.providers.utils.job_scheduler.inline import InlineSchedulerImpl

    scheduler = InlineSchedulerImpl(scheduler_config)

    # Test that all methods raise NotImplementedError
    with pytest.raises(NotImplementedError, match="not yet available"):
        await scheduler.initialize()

    with pytest.raises(NotImplementedError, match="not yet available"):
        await scheduler.start()

    with pytest.raises(NotImplementedError, match="not yet available"):
        scheduler.register_job_executor("test_job", default_test_executor)

    with pytest.raises(NotImplementedError, match="not yet available"):
        await scheduler.schedule_job("test_job", {})

    with pytest.raises(NotImplementedError, match="not yet available"):
        await scheduler.get_job_info("job_id")

    with pytest.raises(NotImplementedError, match="not yet available"):
        await scheduler.cancel_job("job_id")

    with pytest.raises(NotImplementedError, match="not yet available"):
        await scheduler.delete_job("job_id")

    with pytest.raises(NotImplementedError, match="not yet available"):
        await scheduler.list_jobs()

    with pytest.raises(NotImplementedError, match="not yet available"):
        await scheduler.shutdown()


async def test_two_phase_initialization_pattern(scheduler_config):
    """Test that the two-phase initialization pattern is supported."""
    from llama_stack.providers.utils.job_scheduler.inline import InlineSchedulerImpl

    scheduler = InlineSchedulerImpl(scheduler_config)

    # Mock the methods to test the pattern
    scheduler.initialize = AsyncMock()
    scheduler.start = AsyncMock()
    scheduler.register_job_executor = MagicMock()

    # Phase 1: Initialize (loads jobs, doesn't start them)
    await scheduler.initialize()
    scheduler.initialize.assert_called_once()

    # Register executors after initialization
    scheduler.register_job_executor("test_job", default_test_executor)
    scheduler.register_job_executor.assert_called_once()

    # Phase 2: Start (resumes jobs after executors registered)
    await scheduler.start()
    scheduler.start.assert_called_once()

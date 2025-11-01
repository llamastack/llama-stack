# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import tempfile
from pathlib import Path

import pytest

from llama_stack.core.storage.datatypes import KVStoreReference, SqliteKVStoreConfig
from llama_stack.providers.utils.job_scheduler import (
    InlineSchedulerConfig,
    JobStatus,
    scheduler_impl,
)
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


async def test_inline_scheduler_basic(scheduler_config):
    """Test basic scheduler functionality."""
    scheduler = await scheduler_impl(scheduler_config)

    try:
        # Register default executor
        scheduler.register_job_executor("test_job", default_test_executor)

        # Schedule a job
        job_id = await scheduler.schedule_job(
            job_type="test_job",
            job_data={"test": "data"},
            metadata={"user": "test_user"},
        )

        assert job_id is not None
        assert isinstance(job_id, str)

        # Wait a bit for the job to complete
        await asyncio.sleep(0.2)

        # Get job info
        job_info = await scheduler.get_job_info(job_id)
        assert job_info["job_id"] == job_id
        assert job_info["job_type"] == "test_job"
        assert job_info["status"] == JobStatus.COMPLETED.value
        assert job_info["metadata"]["user"] == "test_user"
        assert job_info["progress"] == 1.0
        assert job_info["result"] is not None

    finally:
        await scheduler.shutdown()


async def test_inline_scheduler_list_jobs(scheduler_config):
    """Test listing jobs with filters."""
    scheduler = await scheduler_impl(scheduler_config)

    try:
        # Register executors for different job types
        scheduler.register_job_executor("batch_processing", default_test_executor)
        scheduler.register_job_executor("log_aggregation", default_test_executor)

        # Schedule multiple jobs
        await scheduler.schedule_job(
            job_type="batch_processing",
            job_data={"batch": 1},
        )
        await scheduler.schedule_job(
            job_type="log_aggregation",
            job_data={"logs": []},
        )
        await scheduler.schedule_job(
            job_type="batch_processing",
            job_data={"batch": 2},
        )

        # Wait for jobs to complete
        await asyncio.sleep(0.3)

        # List all jobs
        all_jobs = await scheduler.list_jobs()
        assert len(all_jobs) == 3

        # List jobs by type
        batch_jobs = await scheduler.list_jobs(job_type="batch_processing")
        assert len(batch_jobs) == 2

        log_jobs = await scheduler.list_jobs(job_type="log_aggregation")
        assert len(log_jobs) == 1

        # List jobs by status
        completed_jobs = await scheduler.list_jobs(status=JobStatus.COMPLETED)
        assert len(completed_jobs) == 3

    finally:
        await scheduler.shutdown()


async def test_inline_scheduler_cancel_job(scheduler_config):
    """Test cancelling a job."""
    scheduler = await scheduler_impl(scheduler_config)

    try:
        # Register executor
        scheduler.register_job_executor("long_running_job", default_test_executor)

        # Schedule a job
        job_id = await scheduler.schedule_job(
            job_type="long_running_job",
            job_data={"duration": 10},
        )

        # Try to cancel immediately
        await scheduler.cancel_job(job_id)

        # Wait a bit
        await asyncio.sleep(0.1)

        # Check job status
        job_info = await scheduler.get_job_info(job_id)
        # Job might be CANCELLED or COMPLETED depending on timing
        assert job_info["status"] in [JobStatus.CANCELLED.value, JobStatus.COMPLETED.value]

    finally:
        await scheduler.shutdown()


async def test_inline_scheduler_delete_job(scheduler_config):
    """Test deleting a completed job."""
    scheduler = await scheduler_impl(scheduler_config)

    try:
        # Register executor
        scheduler.register_job_executor("test_job", default_test_executor)

        # Schedule a job
        job_id = await scheduler.schedule_job(
            job_type="test_job",
            job_data={"test": "data"},
        )

        # Wait for completion
        await asyncio.sleep(0.2)

        # Verify job exists
        job_info = await scheduler.get_job_info(job_id)
        assert job_info["status"] == JobStatus.COMPLETED.value

        # Delete the job
        success = await scheduler.delete_job(job_id)
        assert success is True

        # Verify job is deleted
        with pytest.raises(ValueError, match="not found"):
            await scheduler.get_job_info(job_id)

        # Deleting again should return False
        success = await scheduler.delete_job(job_id)
        assert success is False

    finally:
        await scheduler.shutdown()


async def test_inline_scheduler_concurrent_jobs(scheduler_config):
    """Test running multiple jobs concurrently."""
    scheduler = await scheduler_impl(scheduler_config)

    try:
        # Register executor
        scheduler.register_job_executor("test_job", default_test_executor)

        # Schedule multiple jobs
        job_ids = []
        for i in range(5):
            job_id = await scheduler.schedule_job(
                job_type="test_job",
                job_data={"index": i},
            )
            job_ids.append(job_id)

        # Wait for all jobs to complete
        await asyncio.sleep(0.5)

        # Verify all jobs completed
        for job_id in job_ids:
            job_info = await scheduler.get_job_info(job_id)
            assert job_info["status"] == JobStatus.COMPLETED.value

    finally:
        await scheduler.shutdown()


async def test_inline_scheduler_pagination(scheduler_config):
    """Test job listing with pagination."""
    scheduler = await scheduler_impl(scheduler_config)

    try:
        # Register executor
        scheduler.register_job_executor("test_job", default_test_executor)

        # Schedule 10 jobs
        for i in range(10):
            await scheduler.schedule_job(
                job_type="test_job",
                job_data={"index": i},
            )

        # Wait for completion
        await asyncio.sleep(0.5)

        # Test pagination
        page1 = await scheduler.list_jobs(limit=5, offset=0)
        assert len(page1) == 5

        page2 = await scheduler.list_jobs(limit=5, offset=5)
        assert len(page2) == 5

        page3 = await scheduler.list_jobs(limit=5, offset=10)
        assert len(page3) == 0

    finally:
        await scheduler.shutdown()


async def test_inline_scheduler_register_job_executor(scheduler_config):
    """Test registering a job executor."""
    scheduler = await scheduler_impl(scheduler_config)

    try:
        # Define a custom job executor
        async def custom_executor(job_data: dict) -> dict:
            return {"custom": "result", "input": job_data}

        # Register the executor
        scheduler.register_job_executor("custom_job_type", custom_executor)

        # Schedule a job with the custom type
        job_id = await scheduler.schedule_job(
            job_type="custom_job_type",
            job_data={"test": "data"},
        )

        # Wait for job to complete
        await asyncio.sleep(0.2)

        # Verify the custom executor was called
        job_info = await scheduler.get_job_info(job_id)
        assert job_info["status"] == JobStatus.COMPLETED.value
        assert job_info["result"]["custom"] == "result"
        assert job_info["result"]["input"]["test"] == "data"

    finally:
        await scheduler.shutdown()

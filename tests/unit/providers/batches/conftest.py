# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Shared fixtures for batches provider unit tests."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from llama_stack.providers.inline.batches.reference.batches import ReferenceBatchesImpl
from llama_stack.providers.inline.batches.reference.config import ReferenceBatchesImplConfig
from llama_stack.providers.utils.kvstore import kvstore_impl
from llama_stack.providers.utils.kvstore.config import SqliteKVStoreConfig


@pytest.fixture
async def provider():
    """Create a test provider instance with temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_batches.db"
        kvstore_config = SqliteKVStoreConfig(db_path=str(db_path))
        config = ReferenceBatchesImplConfig(kvstore=kvstore_config)

        # Create kvstore and mock APIs
        kvstore = await kvstore_impl(config.kvstore)
        mock_inference = AsyncMock()
        mock_files = AsyncMock()
        mock_models = AsyncMock()

        provider = ReferenceBatchesImpl(config, mock_inference, mock_files, mock_models, kvstore)
        await provider.initialize()

        # unit tests should not require background processing
        provider.process_batches = False

        yield provider

        await provider.shutdown()


@pytest.fixture
def sample_batch_data():
    """Sample batch data for testing."""
    return {
        "input_file_id": "file_abc123",
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "metadata": {"test": "true", "priority": "high"},
    }

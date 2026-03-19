# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock

import pytest

from llama_stack.providers.utils.memory.openai_vector_store_mixin import OpenAIVectorStoreMixin
from llama_stack_api import (
    VectorStoreChunkingStrategyAuto,
)


class MockVectorStoreMixin(OpenAIVectorStoreMixin):
    """Mock implementation of OpenAIVectorStoreMixin for testing."""

    def __init__(self, inference_api, files_api, file_processor_api=None):
        self.inference_api = inference_api
        self.files_api = files_api
        self.file_processor_api = file_processor_api
        self.vector_stores_config = MagicMock()
        self.vector_stores_config.contextual_retrieval_params = MagicMock()


class TestOpenAIVectorStoreMixin:
    """Unit tests for OpenAIVectorStoreMixin."""

    @pytest.fixture
    def mock_files_api(self):
        """Create a mock files API."""
        mock = AsyncMock()
        mock.openai_retrieve_file = AsyncMock()
        mock.openai_retrieve_file.return_value = MagicMock(filename="test.pdf")
        return mock

    @pytest.fixture
    def mock_inference_api(self):
        """Create a mock inference API."""
        return AsyncMock()

    async def test_missing_file_processor_api_raises_runtime_error(
        self, mock_inference_api, mock_files_api
    ):
        """Test that missing file_processor_api raises clear RuntimeError."""
        # Create mixin WITHOUT file_processor_api
        mixin = MockVectorStoreMixin(
            inference_api=mock_inference_api,
            files_api=mock_files_api,
            file_processor_api=None,  # Not configured
        )

        # Mock vector store ID
        vector_store_id = "test_vector_store"
        file_id = "test_file_id"

        # Attempt to add file to vector store should fail with clear error
        with pytest.raises(
            RuntimeError,
            match="FileProcessor API is required for file processing but is not configured",
        ):
            await mixin.openai_add_file_to_vector_store(
                vector_store_id=vector_store_id,
                file_id=file_id,
                chunking_strategy=VectorStoreChunkingStrategyAuto(),
            )

    async def test_file_processor_api_configured_succeeds(
        self, mock_inference_api, mock_files_api
    ):
        """Test that with file_processor_api configured, processing proceeds."""
        # Create mock file processor API
        mock_file_processor_api = AsyncMock()
        mock_file_processor_api.process_file = AsyncMock()
        mock_file_processor_api.process_file.return_value = MagicMock(
            chunks=[], metadata={"processor": "pypdf"}
        )

        # Create mixin WITH file_processor_api
        mixin = MockVectorStoreMixin(
            inference_api=mock_inference_api,
            files_api=mock_files_api,
            file_processor_api=mock_file_processor_api,
        )

        # This should NOT raise the RuntimeError
        # Note: This test will fail later due to other missing dependencies,
        # but we're only testing that the file_processor_api check passes
        vector_store_id = "test_vector_store"
        file_id = "test_file_id"

        # The call should get past the file_processor_api check
        # (will fail later due to missing vector store, but that's expected)
        try:
            await mixin.openai_add_file_to_vector_store(
                vector_store_id=vector_store_id,
                file_id=file_id,
                chunking_strategy=VectorStoreChunkingStrategyAuto(),
            )
        except RuntimeError as e:
            # Should NOT be the file_processor_api error
            assert "FileProcessor API is required" not in str(e)
        except Exception:
            # Other exceptions are fine, we're just testing file_processor_api check
            pass

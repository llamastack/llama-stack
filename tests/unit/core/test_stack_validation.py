# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for Stack validation functions."""

from unittest.mock import AsyncMock

import pytest

from llama_stack.apis.models import ListModelsResponse, Model, ModelType
from llama_stack.core.datatypes import QualifiedModel, StackRunConfig, StorageConfig, VectorStoresConfig
from llama_stack.core.stack import validate_vector_stores_config
from llama_stack.providers.datatypes import Api


class TestVectorStoresValidation:
    async def test_validate_missing_model(self):
        """Test validation fails when model not found."""
        run_config = StackRunConfig(
            image_name="test",
            providers={},
            storage=StorageConfig(backends={}, stores={}),
            vector_stores=VectorStoresConfig(
                default_provider_id="faiss",
                default_embedding_model=QualifiedModel(
                    provider_id="p",
                    model_id="missing",
                ),
            ),
        )
        mock_models = AsyncMock()
        mock_models.list_models.return_value = ListModelsResponse(data=[])

        with pytest.raises(ValueError, match="not found"):
            await validate_vector_stores_config(run_config.vector_stores, {Api.models: mock_models})

    async def test_validate_success(self):
        """Test validation passes with valid model."""
        run_config = StackRunConfig(
            image_name="test",
            providers={},
            storage=StorageConfig(backends={}, stores={}),
            vector_stores=VectorStoresConfig(
                default_provider_id="faiss",
                default_embedding_model=QualifiedModel(
                    provider_id="p",
                    model_id="valid",
                ),
            ),
        )
        mock_models = AsyncMock()
        mock_models.list_models.return_value = ListModelsResponse(
            data=[
                Model(
                    identifier="p/valid",  # Must match provider_id/model_id format
                    model_type=ModelType.embedding,
                    metadata={"embedding_dimension": 768},
                    provider_id="p",
                    provider_resource_id="valid",
                )
            ]
        )

        await validate_vector_stores_config(run_config.vector_stores, {Api.models: mock_models})

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for Stack validation functions."""

from unittest.mock import AsyncMock

import pytest

from llama_stack.core.datatypes import (
    QualifiedModel,
    RewriteQueryParams,
    SafetyConfig,
    StackConfig,
    VectorStoresConfig,
)
from llama_stack.core.stack import validate_safety_config, validate_vector_stores_config
from llama_stack.core.storage.datatypes import ServerStoresConfig, StorageConfig
from llama_stack_api import Api, ListModelsResponse, ListShieldsResponse, Model, ModelType, Shield


class TestVectorStoresValidation:
    async def test_validate_missing_model(self):
        """Test validation fails when model not found."""
        run_config = StackConfig(
            image_name="test",
            providers={},
            storage=StorageConfig(
                backends={},
                stores=ServerStoresConfig(
                    metadata=None,
                    inference=None,
                    conversations=None,
                    prompts=None,
                ),
            ),
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
        run_config = StackConfig(
            image_name="test",
            providers={},
            storage=StorageConfig(
                backends={},
                stores=ServerStoresConfig(
                    metadata=None,
                    inference=None,
                    conversations=None,
                    prompts=None,
                ),
            ),
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

    async def test_validate_rewrite_query_prompt_missing_placeholder(self):
        """Test validation fails when prompt template is missing {query} placeholder."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match=r"prompt must contain \{query\} placeholder"):
            RewriteQueryParams(
                prompt="This prompt has no placeholder",
            )


class TestSafetyConfigValidation:
    async def test_validate_success(self):
        safety_config = SafetyConfig(default_shield_id="shield-1")

        shield = Shield(
            identifier="shield-1",
            provider_id="provider-x",
            provider_resource_id="model-x",
            params={},
        )

        shields_impl = AsyncMock()
        shields_impl.list_shields.return_value = ListShieldsResponse(data=[shield])

        await validate_safety_config(safety_config, {Api.shields: shields_impl, Api.safety: AsyncMock()})

    async def test_validate_wrong_shield_id(self):
        safety_config = SafetyConfig(default_shield_id="wrong-shield-id")

        shields_impl = AsyncMock()
        shields_impl.list_shields.return_value = ListShieldsResponse(
            data=[
                Shield(
                    identifier="shield-1",
                    provider_resource_id="model-x",
                    provider_id="provider-x",
                    params={},
                )
            ]
        )
        with pytest.raises(ValueError, match="wrong-shield-id"):
            await validate_safety_config(safety_config, {Api.shields: shields_impl, Api.safety: AsyncMock()})

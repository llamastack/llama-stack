# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Unit tests for Stack validation functions.
"""

from unittest.mock import AsyncMock

import pytest

from llama_stack.apis.models import Model, ModelType
from llama_stack.core.stack import validate_default_embedding_model
from llama_stack.providers.datatypes import Api


class TestStackValidation:
    """Test Stack validation functions."""

    @pytest.mark.parametrize(
        "models,should_raise",
        [
            ([], False),  # No models
            (
                [
                    Model(
                        identifier="emb1",
                        model_type=ModelType.embedding,
                        metadata={"default_configured": True},
                        provider_id="p",
                        provider_resource_id="emb1",
                    )
                ],
                False,
            ),  # Single default
            (
                [
                    Model(
                        identifier="emb1",
                        model_type=ModelType.embedding,
                        metadata={"default_configured": True},
                        provider_id="p",
                        provider_resource_id="emb1",
                    ),
                    Model(
                        identifier="emb2",
                        model_type=ModelType.embedding,
                        metadata={"default_configured": True},
                        provider_id="p",
                        provider_resource_id="emb2",
                    ),
                ],
                True,
            ),  # Multiple defaults
            (
                [
                    Model(
                        identifier="emb1",
                        model_type=ModelType.embedding,
                        metadata={"default_configured": True},
                        provider_id="p",
                        provider_resource_id="emb1",
                    ),
                    Model(
                        identifier="llm1",
                        model_type=ModelType.llm,
                        metadata={"default_configured": True},
                        provider_id="p",
                        provider_resource_id="llm1",
                    ),
                ],
                False,
            ),  # Ignores non-embedding
        ],
    )
    async def test_validate_default_embedding_model(self, models, should_raise):
        """Test validation with various model configurations."""
        mock_models_impl = AsyncMock()
        mock_models_impl.list_models.return_value = models
        impls = {Api.models: mock_models_impl}

        if should_raise:
            with pytest.raises(ValueError, match="Multiple embedding models marked as default_configured=True"):
                await validate_default_embedding_model(impls)
        else:
            await validate_default_embedding_model(impls)

    async def test_validate_default_embedding_model_no_models_api(self):
        """Test validation when models API is not available."""
        await validate_default_embedding_model({})

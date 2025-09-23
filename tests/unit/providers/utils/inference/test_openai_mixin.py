# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from llama_stack.apis.inference import Model
from llama_stack.apis.models import ModelType
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin


# Test implementation of OpenAIMixin for testing purposes
class OpenAIMixinImpl(OpenAIMixin):
    def __init__(self):
        self.__provider_id__ = "test-provider"

    def get_api_key(self) -> str:
        raise NotImplementedError("This method should be mocked in tests")

    def get_base_url(self) -> str:
        raise NotImplementedError("This method should be mocked in tests")


@pytest.fixture
def mixin():
    """Create a test instance of OpenAIMixin"""
    return OpenAIMixinImpl()


@pytest.fixture
def mock_models():
    """Create multiple mock OpenAI model objects"""
    models = [MagicMock(id=id) for id in ["some-mock-model-id", "another-mock-model-id", "final-mock-model-id"]]
    return models


@pytest.fixture
def mock_client_with_models(mock_models):
    """Create a mock client with models.list() set up to return mock_models"""
    mock_client = MagicMock()

    async def mock_models_list():
        for model in mock_models:
            yield model

    mock_client.models.list.return_value = mock_models_list()
    return mock_client


@pytest.fixture
def mock_client_with_empty_models():
    """Create a mock client with models.list() set up to return empty list"""
    mock_client = MagicMock()

    async def mock_empty_models_list():
        return
        yield  # Make it an async generator but don't yield anything

    mock_client.models.list.return_value = mock_empty_models_list()
    return mock_client


@pytest.fixture
def mock_client_with_exception():
    """Create a mock client with models.list() set up to raise an exception"""
    mock_client = MagicMock()
    mock_client.models.list.side_effect = Exception("API Error")
    return mock_client


@pytest.fixture
def mock_client_context():
    """Fixture that provides a context manager for mocking the OpenAI client"""

    def _mock_client_context(mixin, mock_client):
        return patch.object(type(mixin), "client", new_callable=PropertyMock, return_value=mock_client)

    return _mock_client_context


class TestOpenAIMixinListModels:
    """Test cases for the list_models method"""

    async def test_list_models_success(self, mixin, mock_client_with_models, mock_client_context):
        """Test successful model listing"""
        assert len(mixin._model_cache) == 0

        with mock_client_context(mixin, mock_client_with_models):
            result = await mixin.list_models()

            assert result is not None
            assert len(result) == 3

            model_ids = [model.identifier for model in result]
            assert "some-mock-model-id" in model_ids
            assert "another-mock-model-id" in model_ids
            assert "final-mock-model-id" in model_ids

            for model in result:
                assert model.provider_id == "test-provider"
                assert model.model_type == ModelType.llm
                assert model.provider_resource_id == model.identifier

            assert len(mixin._model_cache) == 3
            for model_id in ["some-mock-model-id", "another-mock-model-id", "final-mock-model-id"]:
                assert model_id in mixin._model_cache
                cached_model = mixin._model_cache[model_id]
                assert cached_model.identifier == model_id
                assert cached_model.provider_resource_id == model_id

    async def test_list_models_empty_response(self, mixin, mock_client_with_empty_models, mock_client_context):
        """Test handling of empty model list"""
        with mock_client_context(mixin, mock_client_with_empty_models):
            result = await mixin.list_models()

            assert result is not None
            assert len(result) == 0
            assert len(mixin._model_cache) == 0


class TestOpenAIMixinCheckModelAvailability:
    """Test cases for the check_model_availability method"""

    async def test_check_model_availability_with_cache(self, mixin, mock_client_with_models, mock_client_context):
        """Test model availability check when cache is populated"""
        with mock_client_context(mixin, mock_client_with_models):
            mock_client_with_models.models.list.assert_not_called()
            await mixin.list_models()
            mock_client_with_models.models.list.assert_called_once()

            assert await mixin.check_model_availability("some-mock-model-id")
            assert await mixin.check_model_availability("another-mock-model-id")
            assert await mixin.check_model_availability("final-mock-model-id")
            assert not await mixin.check_model_availability("non-existent-model")
            mock_client_with_models.models.list.assert_called_once()

    async def test_check_model_availability_without_cache(self, mixin, mock_client_with_models, mock_client_context):
        """Test model availability check when cache is empty (calls list_models)"""
        assert len(mixin._model_cache) == 0

        with mock_client_context(mixin, mock_client_with_models):
            mock_client_with_models.models.list.assert_not_called()
            assert await mixin.check_model_availability("some-mock-model-id")
            mock_client_with_models.models.list.assert_called_once()

            assert len(mixin._model_cache) == 3
            assert "some-mock-model-id" in mixin._model_cache

    async def test_check_model_availability_model_not_found(self, mixin, mock_client_with_models, mock_client_context):
        """Test model availability check for non-existent model"""
        with mock_client_context(mixin, mock_client_with_models):
            mock_client_with_models.models.list.assert_not_called()
            assert not await mixin.check_model_availability("non-existent-model")
            mock_client_with_models.models.list.assert_called_once()

            assert len(mixin._model_cache) == 3


class TestOpenAIMixinCacheBehavior:
    """Test cases for cache behavior and edge cases"""

    async def test_cache_overwrites_on_list_models_call(self, mixin, mock_client_with_models, mock_client_context):
        """Test that calling list_models overwrites existing cache"""
        initial_model = Model(
            provider_id="test-provider",
            provider_resource_id="old-model",
            identifier="old-model",
            model_type=ModelType.llm,
        )
        mixin._model_cache = {"old-model": initial_model}

        with mock_client_context(mixin, mock_client_with_models):
            await mixin.list_models()

            assert len(mixin._model_cache) == 3
            assert "old-model" not in mixin._model_cache
            assert "some-mock-model-id" in mixin._model_cache
            assert "another-mock-model-id" in mixin._model_cache
            assert "final-mock-model-id" in mixin._model_cache

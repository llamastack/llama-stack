# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from unittest.mock import AsyncMock, MagicMock, Mock, PropertyMock, patch

import pytest
from pydantic import BaseModel, Field

from llama_stack.apis.inference import Model, OpenAIUserMessageParam
from llama_stack.apis.models import ModelType
from llama_stack.core.request_headers import request_provider_data_context
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin


class OpenAIMixinImpl(OpenAIMixin):
    __provider_id__: str = "test-provider"

    def get_api_key(self) -> str:
        raise NotImplementedError("This method should be mocked in tests")

    def get_base_url(self) -> str:
        raise NotImplementedError("This method should be mocked in tests")


class OpenAIMixinWithEmbeddingsImpl(OpenAIMixinImpl):
    """Test implementation with embedding model metadata"""

    embedding_model_metadata = {
        "text-embedding-3-small": {"embedding_dimension": 1536, "context_length": 8192},
        "text-embedding-ada-002": {"embedding_dimension": 1536, "context_length": 8192},
    }


@pytest.fixture
def mixin():
    """Create a test instance of OpenAIMixin with mocked model_store"""
    mixin_instance = OpenAIMixinImpl()

    # just enough to satisfy _get_provider_model_id calls
    mock_model_store = MagicMock()
    mock_model = MagicMock()
    mock_model.provider_resource_id = "test-provider-resource-id"
    mock_model_store.get_model = AsyncMock(return_value=mock_model)
    mixin_instance.model_store = mock_model_store

    return mixin_instance


@pytest.fixture
def mixin_with_embeddings():
    """Create a test instance of OpenAIMixin with embedding model metadata"""
    return OpenAIMixinWithEmbeddingsImpl()


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


class TestOpenAIMixinImagePreprocessing:
    """Test cases for image preprocessing functionality"""

    async def test_openai_chat_completion_with_image_preprocessing_enabled(self, mixin):
        """Test that image URLs are converted to base64 when download_images is True"""
        mixin.download_images = True

        message = OpenAIUserMessageParam(
            role="user",
            content=[
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}},
            ],
        )

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.object(type(mixin), "client", new_callable=PropertyMock, return_value=mock_client):
            with patch("llama_stack.providers.utils.inference.openai_mixin.localize_image_content") as mock_localize:
                mock_localize.return_value = (b"fake_image_data", "jpeg")

                await mixin.openai_chat_completion(model="test-model", messages=[message])

            mock_localize.assert_called_once_with("http://example.com/image.jpg")

            mock_client.chat.completions.create.assert_called_once()
            call_args = mock_client.chat.completions.create.call_args
            processed_messages = call_args[1]["messages"]
            assert len(processed_messages) == 1
            content = processed_messages[0]["content"]
            assert len(content) == 2
            assert content[0]["type"] == "text"
            assert content[1]["type"] == "image_url"
            assert content[1]["image_url"]["url"] == "data:image/jpeg;base64,ZmFrZV9pbWFnZV9kYXRh"

    async def test_openai_chat_completion_with_image_preprocessing_disabled(self, mixin):
        """Test that image URLs are not modified when download_images is False"""
        mixin.download_images = False  # explicitly set to False

        message = OpenAIUserMessageParam(
            role="user",
            content=[
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}},
            ],
        )

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.object(type(mixin), "client", new_callable=PropertyMock, return_value=mock_client):
            with patch("llama_stack.providers.utils.inference.openai_mixin.localize_image_content") as mock_localize:
                await mixin.openai_chat_completion(model="test-model", messages=[message])

            mock_localize.assert_not_called()

            mock_client.chat.completions.create.assert_called_once()
            call_args = mock_client.chat.completions.create.call_args
            processed_messages = call_args[1]["messages"]
            assert len(processed_messages) == 1
            content = processed_messages[0]["content"]
            assert len(content) == 2
            assert content[1]["image_url"]["url"] == "http://example.com/image.jpg"


class TestOpenAIMixinEmbeddingModelMetadata:
    """Test cases for embedding_model_metadata attribute functionality"""

    async def test_embedding_model_identified_and_augmented(self, mixin_with_embeddings, mock_client_context):
        """Test that models in embedding_model_metadata are correctly identified as embeddings with metadata"""
        # Create mock models: 1 embedding model and 1 LLM, while there are 2 known embedding models
        mock_embedding_model = MagicMock(id="text-embedding-3-small")
        mock_llm_model = MagicMock(id="gpt-4")
        mock_models = [mock_embedding_model, mock_llm_model]

        mock_client = MagicMock()

        async def mock_models_list():
            for model in mock_models:
                yield model

        mock_client.models.list.return_value = mock_models_list()

        with mock_client_context(mixin_with_embeddings, mock_client):
            result = await mixin_with_embeddings.list_models()

            assert result is not None
            assert len(result) == 2

            # Find the models in the result
            embedding_model = next(m for m in result if m.identifier == "text-embedding-3-small")
            llm_model = next(m for m in result if m.identifier == "gpt-4")

            # Check embedding model
            assert embedding_model.model_type == ModelType.embedding
            assert embedding_model.metadata == {"embedding_dimension": 1536, "context_length": 8192}
            assert embedding_model.provider_id == "test-provider"
            assert embedding_model.provider_resource_id == "text-embedding-3-small"

            # Check LLM model
            assert llm_model.model_type == ModelType.llm
            assert llm_model.metadata == {}  # No metadata for LLMs
            assert llm_model.provider_id == "test-provider"
            assert llm_model.provider_resource_id == "gpt-4"


class TestOpenAIMixinAllowedModels:
    """Test cases for allowed_models filtering functionality"""

    async def test_list_models_with_allowed_models_filter(self, mixin, mock_client_with_models, mock_client_context):
        """Test that list_models filters models based on allowed_models set"""
        mixin.allowed_models = {"some-mock-model-id", "another-mock-model-id"}

        with mock_client_context(mixin, mock_client_with_models):
            result = await mixin.list_models()

            assert result is not None
            assert len(result) == 2

            model_ids = [model.identifier for model in result]
            assert "some-mock-model-id" in model_ids
            assert "another-mock-model-id" in model_ids
            assert "final-mock-model-id" not in model_ids

    async def test_list_models_with_empty_allowed_models(self, mixin, mock_client_with_models, mock_client_context):
        """Test that empty allowed_models set allows all models"""
        assert len(mixin.allowed_models) == 0

        with mock_client_context(mixin, mock_client_with_models):
            result = await mixin.list_models()

            assert result is not None
            assert len(result) == 3  # All models should be included

            model_ids = [model.identifier for model in result]
            assert "some-mock-model-id" in model_ids
            assert "another-mock-model-id" in model_ids
            assert "final-mock-model-id" in model_ids

    async def test_check_model_availability_with_allowed_models(
        self, mixin, mock_client_with_models, mock_client_context
    ):
        """Test that check_model_availability respects allowed_models"""
        mixin.allowed_models = {"final-mock-model-id"}

        with mock_client_context(mixin, mock_client_with_models):
            assert await mixin.check_model_availability("final-mock-model-id")
            assert not await mixin.check_model_availability("some-mock-model-id")
            assert not await mixin.check_model_availability("another-mock-model-id")


class ProviderDataValidator(BaseModel):
    """Validator for provider data in tests"""

    test_api_key: str | None = Field(default=None)


class OpenAIMixinWithProviderData(OpenAIMixinImpl):
    """Test implementation that supports provider data API key field"""

    provider_data_api_key_field: str = "test_api_key"

    def get_api_key(self) -> str:
        return "default-api-key"

    def get_base_url(self):
        return "default-base-url"


class TestOpenAIMixinProviderDataApiKey:
    """Test cases for provider_data_api_key_field functionality"""

    @pytest.fixture
    def mixin_with_provider_data_field(self):
        """Mixin instance with provider_data_api_key_field set"""
        mixin_instance = OpenAIMixinWithProviderData()

        # Mock provider_spec for provider data validation
        mock_provider_spec = MagicMock()
        mock_provider_spec.provider_type = "test-provider-with-data"
        mock_provider_spec.provider_data_validator = (
            "tests.unit.providers.utils.inference.test_openai_mixin.ProviderDataValidator"
        )
        mixin_instance.__provider_spec__ = mock_provider_spec

        return mixin_instance

    @pytest.fixture
    def mixin_with_provider_data_field_and_none_api_key(self, mixin_with_provider_data_field):
        mixin_with_provider_data_field.get_api_key = Mock(return_value=None)
        return mixin_with_provider_data_field

    def test_no_provider_data(self, mixin_with_provider_data_field):
        """Test that client uses config API key when no provider data is available"""
        assert mixin_with_provider_data_field.client.api_key == "default-api-key"

    def test_with_provider_data(self, mixin_with_provider_data_field):
        """Test that provider data API key overrides config API key"""
        with request_provider_data_context(
            {"x-llamastack-provider-data": json.dumps({"test_api_key": "provider-data-key"})}
        ):
            assert mixin_with_provider_data_field.client.api_key == "provider-data-key"

    def test_with_wrong_key(self, mixin_with_provider_data_field):
        """Test fallback to config when provider data doesn't have the required key"""
        with request_provider_data_context({"x-llamastack-provider-data": json.dumps({"wrong_key": "some-value"})}):
            assert mixin_with_provider_data_field.client.api_key == "default-api-key"

    def test_error_when_no_config_and_provider_data_has_wrong_key(
        self, mixin_with_provider_data_field_and_none_api_key
    ):
        """Test that ValueError is raised when provider data exists but doesn't have required key"""
        with request_provider_data_context({"x-llamastack-provider-data": json.dumps({"wrong_key": "some-value"})}):
            with pytest.raises(ValueError, match="API key is not set"):
                _ = mixin_with_provider_data_field_and_none_api_key.client

    def test_error_message_includes_correct_field_names(self, mixin_with_provider_data_field_and_none_api_key):
        """Test that error message includes correct field name and header information"""
        with pytest.raises(ValueError) as exc_info:
            _ = mixin_with_provider_data_field_and_none_api_key.client

        error_message = str(exc_info.value)
        assert "test_api_key" in error_message
        assert "x-llamastack-provider-data" in error_message

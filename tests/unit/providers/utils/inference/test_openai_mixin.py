# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from collections.abc import Iterable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, PropertyMock, patch

import pytest
from pydantic import BaseModel, Field

from llama_stack.apis.inference import Model, OpenAIChatCompletionRequestWithExtraBody, OpenAIUserMessageParam
from llama_stack.apis.models import ModelType
from llama_stack.core.request_headers import request_provider_data_context
from llama_stack.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin


class OpenAIMixinImpl(OpenAIMixin):
    __provider_id__: str = "test-provider"

    def get_api_key(self) -> str:
        return "test-api-key"

    def get_base_url(self) -> str:
        return "http://test-base-url"


class OpenAIMixinWithEmbeddingsImpl(OpenAIMixinImpl):
    """Test implementation with embedding model metadata"""

    embedding_model_metadata: dict[str, dict[str, int]] = {
        "text-embedding-3-small": {"embedding_dimension": 1536, "context_length": 8192},
        "text-embedding-ada-002": {"embedding_dimension": 1536, "context_length": 8192},
    }


class OpenAIMixinWithCustomModelConstruction(OpenAIMixinImpl):
    """Test implementation that uses construct_model_from_identifier to add rerank models"""

    embedding_model_metadata: dict[str, dict[str, int]] = {
        "text-embedding-3-small": {"embedding_dimension": 1536, "context_length": 8192},
        "text-embedding-ada-002": {"embedding_dimension": 1536, "context_length": 8192},
    }

    # Adds rerank models via construct_model_from_identifier
    rerank_model_ids: set[str] = {"rerank-model-1", "rerank-model-2"}

    def construct_model_from_identifier(self, identifier: str) -> Model:
        if identifier in self.rerank_model_ids:
            return Model(
                provider_id=self.__provider_id__,  # type: ignore[attr-defined]
                provider_resource_id=identifier,
                identifier=identifier,
                model_type=ModelType.rerank,
            )
        return super().construct_model_from_identifier(identifier)


@pytest.fixture
def mixin():
    """Create a test instance of OpenAIMixin with mocked model_store"""
    config = RemoteInferenceProviderConfig()
    mixin_instance = OpenAIMixinImpl(config=config)

    # Mock model_store with async methods
    mock_model_store = AsyncMock()
    mock_model = MagicMock()
    mock_model.provider_resource_id = "test-provider-resource-id"
    mock_model_store.get_model = AsyncMock(return_value=mock_model)
    mock_model_store.has_model = AsyncMock(return_value=False)  # Default to False, tests can override
    mixin_instance.model_store = mock_model_store

    return mixin_instance


@pytest.fixture
def mixin_with_embeddings():
    """Create a test instance of OpenAIMixin with embedding model metadata"""
    config = RemoteInferenceProviderConfig()
    return OpenAIMixinWithEmbeddingsImpl(config=config)


@pytest.fixture
def mixin_with_custom_model_construction():
    """Create a test instance using custom construct_model_from_identifier"""
    config = RemoteInferenceProviderConfig()
    return OpenAIMixinWithCustomModelConstruction(config=config)


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


def _assert_models_match_expected(actual_models, expected_models):
    """Verify the models match expected attributes.

    Args:
        actual_models: List of models to verify
        expected_models: Mapping of model identifier to expected attribute values
    """
    for identifier, expected_attrs in expected_models.items():
        model = next(m for m in actual_models if m.identifier == identifier)
        for attr_name, expected_value in expected_attrs.items():
            assert getattr(model, attr_name) == expected_value


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

    async def test_check_model_availability_with_pre_registered_model(
        self, mixin, mock_client_with_models, mock_client_context
    ):
        """Test that check_model_availability returns True for pre-registered models in model_store"""
        # Mock model_store.has_model to return True for a specific model
        mock_model_store = AsyncMock()
        mock_model_store.has_model = AsyncMock(return_value=True)
        mixin.model_store = mock_model_store

        # Test that pre-registered model is found without calling the provider's API
        with mock_client_context(mixin, mock_client_with_models):
            mock_client_with_models.models.list.assert_not_called()
            assert await mixin.check_model_availability("pre-registered-model")
            # Should not call the provider's list_models since model was found in store
            mock_client_with_models.models.list.assert_not_called()
            mock_model_store.has_model.assert_called_once_with("test-provider/pre-registered-model")

    async def test_check_model_availability_fallback_to_provider_when_not_in_store(
        self, mixin, mock_client_with_models, mock_client_context
    ):
        """Test that check_model_availability falls back to provider when model not in store"""
        # Mock model_store.has_model to return False
        mock_model_store = AsyncMock()
        mock_model_store.has_model = AsyncMock(return_value=False)
        mixin.model_store = mock_model_store

        # Test that it falls back to provider's model cache
        with mock_client_context(mixin, mock_client_with_models):
            mock_client_with_models.models.list.assert_not_called()
            assert await mixin.check_model_availability("some-mock-model-id")
            # Should call the provider's list_models since model was not found in store
            mock_client_with_models.models.list.assert_called_once()
            mock_model_store.has_model.assert_called_once_with("test-provider/some-mock-model-id")


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

                params = OpenAIChatCompletionRequestWithExtraBody(model="test-model", messages=[message])
                await mixin.openai_chat_completion(params)

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
                params = OpenAIChatCompletionRequestWithExtraBody(model="test-model", messages=[message])
                await mixin.openai_chat_completion(params)

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

            expected_models = {
                "text-embedding-3-small": {
                    "model_type": ModelType.embedding,
                    "metadata": {"embedding_dimension": 1536, "context_length": 8192},
                    "provider_id": "test-provider",
                    "provider_resource_id": "text-embedding-3-small",
                },
                "gpt-4": {
                    "model_type": ModelType.llm,
                    "metadata": {},
                    "provider_id": "test-provider",
                    "provider_resource_id": "gpt-4",
                },
            }

            _assert_models_match_expected(result, expected_models)


class TestOpenAIMixinCustomModelConstruction:
    """Test cases for mixed model types (LLM, embedding, rerank) through construct_model_from_identifier"""

    async def test_mixed_model_types_identification(self, mixin_with_custom_model_construction, mock_client_context):
        """Test that LLM, embedding, and rerank models are correctly identified with proper types and metadata"""
        # Create mock models: 1 embedding, 1 rerank, 1 LLM
        mock_embedding_model = MagicMock(id="text-embedding-3-small")
        mock_rerank_model = MagicMock(id="rerank-model-1")
        mock_llm_model = MagicMock(id="gpt-4")
        mock_models = [mock_embedding_model, mock_rerank_model, mock_llm_model]

        mock_client = MagicMock()

        async def mock_models_list():
            for model in mock_models:
                yield model

        mock_client.models.list.return_value = mock_models_list()

        with mock_client_context(mixin_with_custom_model_construction, mock_client):
            result = await mixin_with_custom_model_construction.list_models()

            assert result is not None
            assert len(result) == 3

            expected_models = {
                "text-embedding-3-small": {
                    "model_type": ModelType.embedding,
                    "metadata": {"embedding_dimension": 1536, "context_length": 8192},
                    "provider_id": "test-provider",
                    "provider_resource_id": "text-embedding-3-small",
                },
                "rerank-model-1": {
                    "model_type": ModelType.rerank,
                    "metadata": {},
                    "provider_id": "test-provider",
                    "provider_resource_id": "rerank-model-1",
                },
                "gpt-4": {
                    "model_type": ModelType.llm,
                    "metadata": {},
                    "provider_id": "test-provider",
                    "provider_resource_id": "gpt-4",
                },
            }

            _assert_models_match_expected(result, expected_models)


class TestOpenAIMixinAllowedModels:
    """Test cases for allowed_models filtering functionality"""

    async def test_list_models_with_allowed_models_filter(self, mixin, mock_client_with_models, mock_client_context):
        """Test that list_models filters models based on allowed_models"""
        mixin.config.allowed_models = ["some-mock-model-id", "another-mock-model-id"]

        with mock_client_context(mixin, mock_client_with_models):
            result = await mixin.list_models()

            assert result is not None
            assert len(result) == 2

            model_ids = [model.identifier for model in result]
            assert "some-mock-model-id" in model_ids
            assert "another-mock-model-id" in model_ids
            assert "final-mock-model-id" not in model_ids

    async def test_list_models_with_empty_allowed_models(self, mixin, mock_client_with_models, mock_client_context):
        """Test that empty allowed_models allows no models"""
        mixin.config.allowed_models = []

        with mock_client_context(mixin, mock_client_with_models):
            result = await mixin.list_models()

            assert result is not None
            assert len(result) == 0  # No models should be included

    async def test_list_models_with_omitted_allowed_models(self, mixin, mock_client_with_models, mock_client_context):
        """Test that omitted allowed_models allows all models"""
        assert mixin.config.allowed_models is None

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
        mixin.config.allowed_models = ["final-mock-model-id"]

        with mock_client_context(mixin, mock_client_with_models):
            assert await mixin.check_model_availability("final-mock-model-id")
            assert not await mixin.check_model_availability("some-mock-model-id")
            assert not await mixin.check_model_availability("another-mock-model-id")


class TestOpenAIMixinModelRegistration:
    """Test cases for model registration functionality"""

    async def test_register_model_success(self, mixin, mock_client_with_models, mock_client_context):
        """Test successful model registration when model is available"""
        model = Model(
            provider_id="test-provider",
            provider_resource_id="some-mock-model-id",
            identifier="test-model",
            model_type=ModelType.llm,
        )

        with mock_client_context(mixin, mock_client_with_models):
            result = await mixin.register_model(model)

            assert result == model
            assert result.provider_id == "test-provider"
            assert result.provider_resource_id == "some-mock-model-id"
            assert result.identifier == "test-model"
            assert result.model_type == ModelType.llm
            mock_client_with_models.models.list.assert_called_once()

    async def test_register_model_not_available(self, mixin, mock_client_with_models, mock_client_context):
        """Test model registration failure when model is not available from provider"""
        model = Model(
            provider_id="test-provider",
            provider_resource_id="non-existent-model",
            identifier="test-model",
            model_type=ModelType.llm,
        )

        with mock_client_context(mixin, mock_client_with_models):
            with pytest.raises(
                ValueError, match="Model non-existent-model is not available from provider test-provider"
            ):
                await mixin.register_model(model)
            mock_client_with_models.models.list.assert_called_once()

    async def test_register_model_with_allowed_models_filter(self, mixin, mock_client_with_models, mock_client_context):
        """Test model registration with allowed_models filtering"""
        mixin.config.allowed_models = ["some-mock-model-id"]

        # Test with allowed model
        allowed_model = Model(
            provider_id="test-provider",
            provider_resource_id="some-mock-model-id",
            identifier="allowed-model",
            model_type=ModelType.llm,
        )

        # Test with disallowed model
        disallowed_model = Model(
            provider_id="test-provider",
            provider_resource_id="final-mock-model-id",
            identifier="disallowed-model",
            model_type=ModelType.llm,
        )

        with mock_client_context(mixin, mock_client_with_models):
            result = await mixin.register_model(allowed_model)
            assert result == allowed_model
            with pytest.raises(
                ValueError, match="Model final-mock-model-id is not available from provider test-provider"
            ):
                await mixin.register_model(disallowed_model)
            mock_client_with_models.models.list.assert_called_once()

    async def test_register_embedding_model(self, mixin_with_embeddings, mock_client_context):
        """Test registration of embedding models with metadata"""
        mock_embedding_model = MagicMock(id="text-embedding-3-small")
        mock_models = [mock_embedding_model]

        mock_client = MagicMock()

        async def mock_models_list():
            for model in mock_models:
                yield model

        mock_client.models.list.return_value = mock_models_list()

        embedding_model = Model(
            provider_id="test-provider",
            provider_resource_id="text-embedding-3-small",
            identifier="embedding-test",
            model_type=ModelType.embedding,
        )

        with mock_client_context(mixin_with_embeddings, mock_client):
            result = await mixin_with_embeddings.register_model(embedding_model)
            assert result == embedding_model
            assert result.model_type == ModelType.embedding

    async def test_unregister_model(self, mixin):
        """Test model unregistration (should be no-op)"""
        # unregister_model should not raise any exceptions and return None
        result = await mixin.unregister_model("any-model-id")
        assert result is None

    async def test_should_refresh_models(self, mixin):
        """Test should_refresh_models method returns config value"""
        # Default config has refresh_models=False
        result = await mixin.should_refresh_models()
        assert result is False

        config_with_refresh = RemoteInferenceProviderConfig(refresh_models=True)
        mixin_with_refresh = OpenAIMixinImpl(config=config_with_refresh)
        result_with_refresh = await mixin_with_refresh.should_refresh_models()
        assert result_with_refresh is True

    async def test_register_model_error_propagation(self, mixin, mock_client_with_exception, mock_client_context):
        """Test that errors from provider API are properly propagated during registration"""
        model = Model(
            provider_id="test-provider",
            provider_resource_id="some-model",
            identifier="test-model",
            model_type=ModelType.llm,
        )

        with mock_client_context(mixin, mock_client_with_exception):
            # The exception from the API should be propagated
            with pytest.raises(Exception, match="API Error"):
                await mixin.register_model(model)


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


class CustomListProviderModelIdsImplementation(OpenAIMixinImpl):
    """Test implementation with custom list_provider_model_ids override"""

    custom_model_ids: Any

    async def list_provider_model_ids(self) -> Iterable[str]:
        """Return custom model IDs list"""
        return self.custom_model_ids


class TestOpenAIMixinCustomListProviderModelIds:
    """Test cases for custom list_provider_model_ids() implementation functionality"""

    @pytest.fixture
    def custom_model_ids_list(self):
        """Create a list of custom model ID strings"""
        return ["custom-model-1", "custom-model-2", "custom-embedding"]

    @pytest.fixture
    def config(self):
        """Create RemoteInferenceProviderConfig instance"""
        return RemoteInferenceProviderConfig()

    @pytest.fixture
    def adapter(self, custom_model_ids_list, config):
        """Create mixin instance with custom list_provider_model_ids implementation"""
        mixin = CustomListProviderModelIdsImplementation(config=config, custom_model_ids=custom_model_ids_list)
        mixin.embedding_model_metadata = {"custom-embedding": {"embedding_dimension": 768, "context_length": 512}}
        return mixin

    async def test_is_used(self, adapter, custom_model_ids_list):
        """Test that custom list_provider_model_ids() implementation is used instead of client.models.list()"""
        result = await adapter.list_models()

        assert result is not None
        assert len(result) == 3

        assert set(custom_model_ids_list) == {m.identifier for m in result}

    async def test_populates_cache(self, adapter, custom_model_ids_list):
        """Test that custom list_provider_model_ids() results are cached"""
        assert len(adapter._model_cache) == 0

        await adapter.list_models()

        assert set(custom_model_ids_list) == set(adapter._model_cache.keys())

    async def test_respects_allowed_models(self, config):
        """Test that custom list_provider_model_ids() respects allowed_models filtering"""
        mixin = CustomListProviderModelIdsImplementation(
            config=config, custom_model_ids=["model-1", "model-2", "model-3"]
        )
        mixin.config.allowed_models = ["model-1"]

        result = await mixin.list_models()

        assert result is not None
        assert len(result) == 1
        assert result[0].identifier == "model-1"

    async def test_with_empty_list(self, config):
        """Test that custom list_provider_model_ids() handles empty list correctly"""
        mixin = CustomListProviderModelIdsImplementation(config=config, custom_model_ids=[])

        result = await mixin.list_models()

        assert result is not None
        assert len(result) == 0
        assert len(mixin._model_cache) == 0

    async def test_wrong_type_raises_error(self, config):
        """Test that list_provider_model_ids() returning unhashable items results in an error"""
        mixin = CustomListProviderModelIdsImplementation(
            config=config, custom_model_ids=["valid-model", ["nested", "list"]]
        )
        with pytest.raises(Exception, match="is not a string"):
            await mixin.list_models()

        mixin = CustomListProviderModelIdsImplementation(
            config=config, custom_model_ids=[{"key": "value"}, "valid-model"]
        )
        with pytest.raises(Exception, match="is not a string"):
            await mixin.list_models()

        mixin = CustomListProviderModelIdsImplementation(config=config, custom_model_ids=["valid-model", 42.0])
        with pytest.raises(Exception, match="is not a string"):
            await mixin.list_models()

        mixin = CustomListProviderModelIdsImplementation(config=config, custom_model_ids=[None])
        with pytest.raises(Exception, match="is not a string"):
            await mixin.list_models()

    async def test_non_iterable_raises_error(self, config):
        """Test that list_provider_model_ids() returning non-iterable type raises error"""
        mixin = CustomListProviderModelIdsImplementation(config=config, custom_model_ids=42)

        with pytest.raises(
            TypeError,
            match=r"Failed to list models: CustomListProviderModelIdsImplementation\.list_provider_model_ids\(\) must return an iterable.*but returned int",
        ):
            await mixin.list_models()

    async def test_accepts_various_iterables(self, config):
        """Test that list_provider_model_ids() accepts tuples, sets, generators, etc."""

        tuples = CustomListProviderModelIdsImplementation(
            config=config, custom_model_ids=("model-1", "model-2", "model-3")
        )
        result = await tuples.list_models()
        assert result is not None
        assert len(result) == 3

        class GeneratorAdapter(OpenAIMixinImpl):
            async def list_provider_model_ids(self) -> Iterable[str]:
                def gen():
                    yield "gen-model-1"
                    yield "gen-model-2"

                return gen()

        mixin = GeneratorAdapter(config=config)
        result = await mixin.list_models()
        assert result is not None
        assert len(result) == 2

        sets = CustomListProviderModelIdsImplementation(config=config, custom_model_ids={"set-model-1", "set-model-2"})
        result = await sets.list_models()
        assert result is not None
        assert len(result) == 2


class TestOpenAIMixinProviderDataApiKey:
    """Test cases for provider_data_api_key_field functionality"""

    @pytest.fixture
    def mixin_with_provider_data_field(self):
        """Mixin instance with provider_data_api_key_field set"""
        config = RemoteInferenceProviderConfig()
        mixin_instance = OpenAIMixinWithProviderData(config=config)

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
            with pytest.raises(ValueError, match="API key not provided"):
                _ = mixin_with_provider_data_field_and_none_api_key.client

    def test_error_message_includes_correct_field_names(self, mixin_with_provider_data_field_and_none_api_key):
        """Test that error message includes correct field name and header information"""
        with pytest.raises(ValueError) as exc_info:
            _ = mixin_with_provider_data_field_and_none_api_key.client

        error_message = str(exc_info.value)
        assert "test_api_key" in error_message
        assert "x-llamastack-provider-data" in error_message

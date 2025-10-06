# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from openai import NOT_GIVEN, AsyncOpenAI
from openai.types.model import Model as OpenAIModel

# Import the real Pydantic response types instead of using Mocks
from llama_stack.apis.inference import (
    OpenAIAssistantMessageParam,
    OpenAIChatCompletion,
    OpenAIChoice,
    OpenAICompletion,
    OpenAIEmbeddingData,
    OpenAIEmbeddingsResponse,
    OpenAIEmbeddingUsage,
)
from llama_stack.testing.inference_recorder import (
    InferenceMode,
    ResponseStorage,
    inference_recording,
    normalize_request,
)


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for test recordings."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def real_openai_chat_response():
    """Real OpenAI chat completion response using proper Pydantic objects."""
    return OpenAIChatCompletion(
        id="chatcmpl-test123",
        choices=[
            OpenAIChoice(
                index=0,
                message=OpenAIAssistantMessageParam(
                    role="assistant", content="Hello! I'm doing well, thank you for asking."
                ),
                finish_reason="stop",
            )
        ],
        created=1234567890,
        model="llama3.2:3b",
    )


@pytest.fixture
def real_embeddings_response():
    """Real OpenAI embeddings response using proper Pydantic objects."""
    return OpenAIEmbeddingsResponse(
        object="list",
        data=[
            OpenAIEmbeddingData(object="embedding", embedding=[0.1, 0.2, 0.3], index=0),
            OpenAIEmbeddingData(object="embedding", embedding=[0.4, 0.5, 0.6], index=1),
        ],
        model="nomic-embed-text",
        usage=OpenAIEmbeddingUsage(prompt_tokens=6, total_tokens=6),
    )


class TestInferenceRecording:
    """Test the inference recording system."""

    def test_request_normalization(self):
        """Test that request normalization produces consistent hashes."""
        # Test basic normalization
        hash1 = normalize_request(
            "POST",
            "http://localhost:11434/v1/chat/completions",
            {},
            {"model": "llama3.2:3b", "messages": [{"role": "user", "content": "Hello world"}], "temperature": 0.7},
        )

        # Same request should produce same hash
        hash2 = normalize_request(
            "POST",
            "http://localhost:11434/v1/chat/completions",
            {},
            {"model": "llama3.2:3b", "messages": [{"role": "user", "content": "Hello world"}], "temperature": 0.7},
        )

        assert hash1 == hash2

        # Different content should produce different hash
        hash3 = normalize_request(
            "POST",
            "http://localhost:11434/v1/chat/completions",
            {},
            {
                "model": "llama3.2:3b",
                "messages": [{"role": "user", "content": "Different message"}],
                "temperature": 0.7,
            },
        )

        assert hash1 != hash3

    def test_request_normalization_edge_cases(self):
        """Test request normalization is precise about request content."""
        # Test that different whitespace produces different hashes (no normalization)
        hash1 = normalize_request(
            "POST",
            "http://test/v1/chat/completions",
            {},
            {"messages": [{"role": "user", "content": "Hello   world\n\n"}]},
        )
        hash2 = normalize_request(
            "POST", "http://test/v1/chat/completions", {}, {"messages": [{"role": "user", "content": "Hello world"}]}
        )
        assert hash1 != hash2  # Different whitespace should produce different hashes

        # Test that different float precision produces different hashes (no rounding)
        hash3 = normalize_request("POST", "http://test/v1/chat/completions", {}, {"temperature": 0.7000001})
        hash4 = normalize_request("POST", "http://test/v1/chat/completions", {}, {"temperature": 0.7})
        assert hash3 != hash4  # Different precision should produce different hashes

    def test_response_storage(self, temp_storage_dir):
        """Test the ResponseStorage class."""
        temp_storage_dir = temp_storage_dir / "test_response_storage"
        storage = ResponseStorage(temp_storage_dir)

        # Test storing and retrieving a recording
        request_hash = "test_hash_123"
        request_data = {
            "method": "POST",
            "url": "http://localhost:11434/v1/chat/completions",
            "endpoint": "/v1/chat/completions",
            "model": "llama3.2:3b",
        }
        response_data = {"body": {"content": "test response"}, "is_streaming": False}

        storage.store_recording(request_hash, request_data, response_data)

        # Verify file storage and retrieval
        retrieved = storage.find_recording(request_hash)
        assert retrieved is not None
        assert retrieved["request"]["model"] == "llama3.2:3b"
        assert retrieved["response"]["body"]["content"] == "test response"

    async def test_recording_mode(self, temp_storage_dir, real_openai_chat_response):
        """Test that recording mode captures and stores responses."""
        temp_storage_dir = temp_storage_dir / "test_recording_mode"
        with inference_recording(mode=InferenceMode.RECORD, storage_dir=str(temp_storage_dir)):
            client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")
            client.chat.completions._post = AsyncMock(return_value=real_openai_chat_response)

            response = await client.chat.completions.create(
                model="llama3.2:3b",
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                temperature=0.7,
                max_tokens=50,
                user=NOT_GIVEN,
            )

            # Verify the response was returned correctly
            assert response.choices[0].message.content == "Hello! I'm doing well, thank you for asking."
            client.chat.completions._post.assert_called_once()

        # Verify recording was stored
        storage = ResponseStorage(temp_storage_dir)
        dir = storage._get_test_dir()
        assert dir.exists()

    async def test_replay_mode(self, temp_storage_dir, real_openai_chat_response):
        """Test that replay mode returns stored responses without making real calls."""
        temp_storage_dir = temp_storage_dir / "test_replay_mode"
        # First, record a response
        with inference_recording(mode=InferenceMode.RECORD, storage_dir=str(temp_storage_dir)):
            client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")
            client.chat.completions._post = AsyncMock(return_value=real_openai_chat_response)

            response = await client.chat.completions.create(
                model="llama3.2:3b",
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                temperature=0.7,
                max_tokens=50,
                user=NOT_GIVEN,
            )
            client.chat.completions._post.assert_called_once()

        # Now test replay mode - should not call the original method
        with inference_recording(mode=InferenceMode.REPLAY, storage_dir=str(temp_storage_dir)):
            client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")
            client.chat.completions._post = AsyncMock(return_value=real_openai_chat_response)

            response = await client.chat.completions.create(
                model="llama3.2:3b",
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                temperature=0.7,
                max_tokens=50,
            )

            # Verify we got the recorded response
            assert response.choices[0].message.content == "Hello! I'm doing well, thank you for asking."

            # Verify the original method was NOT called
            client.chat.completions._post.assert_not_called()

    async def test_replay_mode_models(self, temp_storage_dir):
        """Test that replay mode returns stored responses without making real model listing calls."""

        async def _async_iterator(models):
            for model in models:
                yield model

        models = [
            OpenAIModel(id="foo", created=1, object="model", owned_by="test"),
            OpenAIModel(id="bar", created=2, object="model", owned_by="test"),
        ]

        expected_ids = {m.id for m in models}

        temp_storage_dir = temp_storage_dir / "test_replay_mode_models"

        # baseline - mock works without recording
        client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")
        client.models._get_api_list = Mock(return_value=_async_iterator(models))
        assert {m.id async for m in client.models.list()} == expected_ids
        client.models._get_api_list.assert_called_once()

        # record the call
        with inference_recording(mode=InferenceMode.RECORD, storage_dir=temp_storage_dir):
            client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")
            client.models._get_api_list = Mock(return_value=_async_iterator(models))
            assert {m.id async for m in client.models.list()} == expected_ids
            client.models._get_api_list.assert_called_once()

        # replay the call
        with inference_recording(mode=InferenceMode.REPLAY, storage_dir=temp_storage_dir):
            client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")
            client.models._get_api_list = Mock(return_value=_async_iterator(models))
            assert {m.id async for m in client.models.list()} == expected_ids
            client.models._get_api_list.assert_not_called()

    async def test_replay_missing_recording(self, temp_storage_dir):
        """Test that replay mode fails when no recording is found."""
        temp_storage_dir = temp_storage_dir / "test_replay_missing_recording"
        with patch("openai.resources.chat.completions.AsyncCompletions.create"):
            with inference_recording(mode=InferenceMode.REPLAY, storage_dir=str(temp_storage_dir)):
                client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")

                with pytest.raises(RuntimeError, match="No recorded response found"):
                    await client.chat.completions.create(
                        model="llama3.2:3b", messages=[{"role": "user", "content": "This was never recorded"}]
                    )

    async def test_embeddings_recording(self, temp_storage_dir, real_embeddings_response):
        """Test recording and replay of embeddings calls."""

        # baseline - mock works without recording
        client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")
        client.embeddings._post = AsyncMock(return_value=real_embeddings_response)
        response = await client.embeddings.create(
            model=real_embeddings_response.model,
            input=["Hello world", "Test embedding"],
            encoding_format=NOT_GIVEN,
        )
        assert len(response.data) == 2
        assert response.data[0].embedding == [0.1, 0.2, 0.3]
        client.embeddings._post.assert_called_once()

        temp_storage_dir = temp_storage_dir / "test_embeddings_recording"
        # Record
        with inference_recording(mode=InferenceMode.RECORD, storage_dir=str(temp_storage_dir)):
            client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")
            client.embeddings._post = AsyncMock(return_value=real_embeddings_response)

            response = await client.embeddings.create(
                model=real_embeddings_response.model,
                input=["Hello world", "Test embedding"],
                encoding_format=NOT_GIVEN,
                dimensions=NOT_GIVEN,
                user=NOT_GIVEN,
            )

            assert len(response.data) == 2

        # Replay
        with inference_recording(mode=InferenceMode.REPLAY, storage_dir=str(temp_storage_dir)):
            client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")
            client.embeddings._post = AsyncMock(return_value=real_embeddings_response)

            response = await client.embeddings.create(
                model=real_embeddings_response.model,
                input=["Hello world", "Test embedding"],
            )

            # Verify we got the recorded response
            assert len(response.data) == 2
            assert response.data[0].embedding == [0.1, 0.2, 0.3]

            # Verify original method was not called
            client.embeddings._post.assert_not_called()

    async def test_completions_recording(self, temp_storage_dir):
        real_completions_response = OpenAICompletion(
            id="test_completion",
            object="text_completion",
            created=1234567890,
            model="llama3.2:3b",
            choices=[
                {
                    "text": "Hello! I'm doing well, thank you for asking.",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
        )

        temp_storage_dir = temp_storage_dir / "test_completions_recording"

        # baseline - mock works without recording
        client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")
        client.completions._post = AsyncMock(return_value=real_completions_response)
        response = await client.completions.create(
            model=real_completions_response.model,
            prompt="Hello, how are you?",
            temperature=0.7,
            max_tokens=50,
            user=NOT_GIVEN,
        )
        assert response.choices[0].text == real_completions_response.choices[0].text
        client.completions._post.assert_called_once()

        # Record
        with inference_recording(mode=InferenceMode.RECORD, storage_dir=str(temp_storage_dir)):
            client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")
            client.completions._post = AsyncMock(return_value=real_completions_response)

            response = await client.completions.create(
                model=real_completions_response.model,
                prompt="Hello, how are you?",
                temperature=0.7,
                max_tokens=50,
                user=NOT_GIVEN,
            )

            assert response.choices[0].text == real_completions_response.choices[0].text
            client.completions._post.assert_called_once()

        # Replay
        with inference_recording(mode=InferenceMode.REPLAY, storage_dir=str(temp_storage_dir)):
            client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")
            client.completions._post = AsyncMock(return_value=real_completions_response)
            response = await client.completions.create(
                model=real_completions_response.model,
                prompt="Hello, how are you?",
                temperature=0.7,
                max_tokens=50,
            )
            assert response.choices[0].text == real_completions_response.choices[0].text
            client.completions._post.assert_not_called()

    async def test_live_mode(self, real_openai_chat_response):
        """Test that live mode passes through to original methods."""

        async def mock_create(*args, **kwargs):
            return real_openai_chat_response

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=mock_create):
            with inference_recording(mode=InferenceMode.LIVE, storage_dir="foo"):
                client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")

                response = await client.chat.completions.create(
                    model="llama3.2:3b", messages=[{"role": "user", "content": "Hello"}]
                )

                # Verify the response was returned
                assert response.choices[0].message.content == "Hello! I'm doing well, thank you for asking."

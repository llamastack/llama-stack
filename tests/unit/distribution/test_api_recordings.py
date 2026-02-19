# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from openai import AsyncOpenAI, NotFoundError

from llama_stack.testing.api_recorder import (
    APIRecordingMode,
    ResponseStorage,
    api_recording,
    normalize_inference_request,
)

# Import the real Pydantic response types instead of using Mocks
from llama_stack_api import (
    OpenAIChatCompletion,
    OpenAIChatCompletionResponseMessage,
    OpenAIChoice,
    OpenAIEmbeddingData,
    OpenAIEmbeddingsResponse,
    OpenAIEmbeddingUsage,
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
                message=OpenAIChatCompletionResponseMessage(
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
        hash1 = normalize_inference_request(
            "POST",
            "http://localhost:11434/v1/chat/completions",
            {},
            {"model": "llama3.2:3b", "messages": [{"role": "user", "content": "Hello world"}], "temperature": 0.7},
        )

        # Same request should produce same hash
        hash2 = normalize_inference_request(
            "POST",
            "http://localhost:11434/v1/chat/completions",
            {},
            {"model": "llama3.2:3b", "messages": [{"role": "user", "content": "Hello world"}], "temperature": 0.7},
        )

        assert hash1 == hash2

        # Different content should produce different hash
        hash3 = normalize_inference_request(
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
        hash1 = normalize_inference_request(
            "POST",
            "http://test/v1/chat/completions",
            {},
            {"messages": [{"role": "user", "content": "Hello   world\n\n"}]},
        )
        hash2 = normalize_inference_request(
            "POST", "http://test/v1/chat/completions", {}, {"messages": [{"role": "user", "content": "Hello world"}]}
        )
        assert hash1 != hash2  # Different whitespace should produce different hashes

        # Test that different float precision produces different hashes (no rounding)
        hash3 = normalize_inference_request("POST", "http://test/v1/chat/completions", {}, {"temperature": 0.7000001})
        hash4 = normalize_inference_request("POST", "http://test/v1/chat/completions", {}, {"temperature": 0.7})
        assert hash3 == hash4  # Small float precision differences should normalize to the same hash

        # String-embedded decimals with excessive precision should also normalize.
        body_with_precise_scores = {
            "messages": [
                {
                    "role": "tool",
                    "content": "score: 0.7472640164649847",
                }
            ]
        }
        body_with_precise_scores_variation = {
            "messages": [
                {
                    "role": "tool",
                    "content": "score: 0.74726414959878",
                }
            ]
        }
        hash5 = normalize_inference_request("POST", "http://test/v1/chat/completions", {}, body_with_precise_scores)
        hash6 = normalize_inference_request(
            "POST", "http://test/v1/chat/completions", {}, body_with_precise_scores_variation
        )
        assert hash5 == hash6

        body_with_close_scores = {
            "messages": [
                {
                    "role": "tool",
                    "content": "score: 0.662477492560699",
                }
            ]
        }
        body_with_close_scores_variation = {
            "messages": [
                {
                    "role": "tool",
                    "content": "score: 0.6624775971970099",
                }
            ]
        }
        hash7 = normalize_inference_request("POST", "http://test/v1/chat/completions", {}, body_with_close_scores)
        hash8 = normalize_inference_request(
            "POST", "http://test/v1/chat/completions", {}, body_with_close_scores_variation
        )
        assert hash7 == hash8

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

        async def mock_create(*args, **kwargs):
            return real_openai_chat_response

        temp_storage_dir = temp_storage_dir / "test_recording_mode"
        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=mock_create):
            with api_recording(mode=APIRecordingMode.RECORD, storage_dir=str(temp_storage_dir)):
                client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")

                response = await client.chat.completions.create(
                    model="llama3.2:3b",
                    messages=[{"role": "user", "content": "Hello, how are you?"}],
                    temperature=0.7,
                    max_tokens=50,
                )

                # Verify the response was returned correctly
                assert response.choices[0].message.content == "Hello! I'm doing well, thank you for asking."

        # Verify recording was stored
        storage = ResponseStorage(temp_storage_dir)
        assert storage._get_test_dir().exists()

    async def test_replay_mode(self, temp_storage_dir, real_openai_chat_response):
        """Test that replay mode returns stored responses without making real calls."""

        async def mock_create(*args, **kwargs):
            return real_openai_chat_response

        temp_storage_dir = temp_storage_dir / "test_replay_mode"
        # First, record a response
        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=mock_create):
            with api_recording(mode=APIRecordingMode.RECORD, storage_dir=str(temp_storage_dir)):
                client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")

                response = await client.chat.completions.create(
                    model="llama3.2:3b",
                    messages=[{"role": "user", "content": "Hello, how are you?"}],
                    temperature=0.7,
                    max_tokens=50,
                )

        # Now test replay mode - should not call the original method
        with patch("openai.resources.chat.completions.AsyncCompletions.create") as mock_create_patch:
            with api_recording(mode=APIRecordingMode.REPLAY, storage_dir=str(temp_storage_dir)):
                client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")

                response = await client.chat.completions.create(
                    model="llama3.2:3b",
                    messages=[{"role": "user", "content": "Hello, how are you?"}],
                    temperature=0.7,
                    max_tokens=50,
                )

                # Verify we got the recorded response
                assert response.choices[0].message.content == "Hello! I'm doing well, thank you for asking."

                # Verify the original method was NOT called
                mock_create_patch.assert_not_called()

    async def test_replay_missing_recording(self, temp_storage_dir):
        """Test that replay mode fails when no recording is found."""
        temp_storage_dir = temp_storage_dir / "test_replay_missing_recording"
        with patch("openai.resources.chat.completions.AsyncCompletions.create"):
            with api_recording(mode=APIRecordingMode.REPLAY, storage_dir=str(temp_storage_dir)):
                client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")

                with pytest.raises(RuntimeError, match="Recording not found"):
                    await client.chat.completions.create(
                        model="llama3.2:3b", messages=[{"role": "user", "content": "This was never recorded"}]
                    )

    async def test_embeddings_recording(self, temp_storage_dir, real_embeddings_response):
        """Test recording and replay of embeddings calls."""

        async def mock_create(*args, **kwargs):
            return real_embeddings_response

        temp_storage_dir = temp_storage_dir / "test_embeddings_recording"
        # Record
        with patch("openai.resources.embeddings.AsyncEmbeddings.create", side_effect=mock_create):
            with api_recording(mode=APIRecordingMode.RECORD, storage_dir=str(temp_storage_dir)):
                client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")

                response = await client.embeddings.create(
                    model="nomic-embed-text", input=["Hello world", "Test embedding"]
                )

                assert len(response.data) == 2

        # Replay
        with patch("openai.resources.embeddings.AsyncEmbeddings.create") as mock_create_patch:
            with api_recording(mode=APIRecordingMode.REPLAY, storage_dir=str(temp_storage_dir)):
                client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")

                response = await client.embeddings.create(
                    model="nomic-embed-text", input=["Hello world", "Test embedding"]
                )

                # Verify we got the recorded response
                assert len(response.data) == 2
                assert response.data[0].embedding == [0.1, 0.2, 0.3]

                # Verify original method was not called
                mock_create_patch.assert_not_called()

    async def test_live_mode(self, real_openai_chat_response):
        """Test that live mode passes through to original methods."""

        async def mock_create(*args, **kwargs):
            return real_openai_chat_response

        with patch("openai.resources.chat.completions.AsyncCompletions.create", side_effect=mock_create):
            with api_recording(mode=APIRecordingMode.LIVE, storage_dir="foo"):
                client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")

                response = await client.chat.completions.create(
                    model="llama3.2:3b", messages=[{"role": "user", "content": "Hello"}]
                )

                # Verify the response was returned
                assert response.choices[0].message.content == "Hello! I'm doing well, thank you for asking."


class TestExceptionRecordingReplay:
    """Test exception recording/replay in the api_recorder monkey-patching layer.

    Integration tests use record/replay to avoid live API calls. When a provider
    raises an error during recording, we need to:
    1. Serialize the exception and store it alongside normal recordings
    2. Re-raise it so the recording run still sees the original error
    3. On replay, reconstruct the same exception type from the stored data
       so tests that catch specific SDK exceptions (e.g. ``except NotFoundError``)
       continue to work without a live connection
    """

    async def test_record_exception_stores_serialized_error_and_reraises(self, temp_storage_dir):
        """Verify that record mode both persists the exception to disk AND re-raises it.

        Re-raising is important: without it, a recording run would silently swallow
        provider errors, producing a recording file but hiding the failure from the
        developer running in record mode.
        """
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        response = httpx.Response(404, json={"error": {"code": "not_found"}}, request=request)
        exc_to_raise = NotFoundError(
            message="Batch batch-xyz not found",
            response=response,
            body={"error": {"code": "not_found"}},
        )

        async def mock_create_raises(*args, **kwargs):
            raise exc_to_raise

        temp_storage_dir = temp_storage_dir / "test_exception_record"
        with patch(
            "openai.resources.chat.completions.AsyncCompletions.create",
            side_effect=mock_create_raises,
        ):
            with api_recording(mode=APIRecordingMode.RECORD, storage_dir=str(temp_storage_dir)):
                client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")

                with pytest.raises(NotFoundError) as exc_info:
                    await client.chat.completions.create(
                        model="llama3.2:3b",
                        messages=[{"role": "user", "content": "Hello"}],
                    )

                assert exc_info.value.status_code == 404
                assert "not found" in str(exc_info.value).lower()

        # Verify recording was stored with exception data
        storage = ResponseStorage(temp_storage_dir)
        recordings_dir = storage.base_dir / "recordings"
        assert recordings_dir.exists()
        recording_files = list(recordings_dir.glob("*.json"))
        assert len(recording_files) >= 1
        with open(recording_files[0]) as f:
            data = json.load(f)
        assert data["response"]["is_exception"] is True
        assert "exception_data" in data["response"]
        assert data["response"]["exception_data"]["category"] == "provider_sdk"
        assert data["response"]["exception_data"]["provider"] == "openai"
        assert data["response"]["exception_data"]["status_code"] == 404

    async def test_replay_recorded_exception_raises_same_type(self, temp_storage_dir):
        """Verify the full record-then-replay cycle preserves the exception type.

        This is the core guarantee: a test that does ``with pytest.raises(NotFoundError)``
        must pass identically in both record and replay modes. The test records an
        exception, then replays it and asserts:
        - The reconstructed exception is the same SDK type (NotFoundError, not Exception)
        - Status code and message are preserved
        - The original client method is never called during replay
        """
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        response = httpx.Response(404, json={"error": {"code": "not_found"}}, request=request)
        exc_to_raise = NotFoundError(
            message="Batch batch-xyz not found",
            response=response,
            body={"error": {"code": "not_found"}},
        )

        async def mock_create_raises(*args, **kwargs):
            raise exc_to_raise

        temp_storage_dir = temp_storage_dir / "test_exception_replay"
        # Record first
        with patch(
            "openai.resources.chat.completions.AsyncCompletions.create",
            side_effect=mock_create_raises,
        ):
            with api_recording(mode=APIRecordingMode.RECORD, storage_dir=str(temp_storage_dir)):
                client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")
                with pytest.raises(NotFoundError):
                    await client.chat.completions.create(
                        model="llama3.2:3b",
                        messages=[{"role": "user", "content": "Hello"}],
                    )

        # Replay - should raise same exception type without calling original
        with patch("openai.resources.chat.completions.AsyncCompletions.create") as mock_create:
            with api_recording(mode=APIRecordingMode.REPLAY, storage_dir=str(temp_storage_dir)):
                client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")

                with pytest.raises(NotFoundError) as exc_info:
                    await client.chat.completions.create(
                        model="llama3.2:3b",
                        messages=[{"role": "user", "content": "Hello"}],
                    )

                mock_create.assert_not_called()
                assert exc_info.value.status_code == 404
                assert "not found" in str(exc_info.value).lower()

    async def test_replay_legacy_exception_format_raises_generic(self, temp_storage_dir):
        """Verify backwards compatibility with recordings made before exception_data was added.

        Older recordings store only ``exception_message`` (a plain string) without the
        structured ``exception_data`` dict. The replay path must handle this gracefully
        by raising a generic ``Exception`` with the original message, rather than
        crashing on a missing key.
        """
        temp_storage_dir = temp_storage_dir / "test_legacy_exception"
        recordings_dir = temp_storage_dir / "recordings"
        recordings_dir.mkdir(parents=True, exist_ok=True)

        # URL must match what AsyncOpenAI(base_url="http://localhost:11434/v1") produces
        # for /v1/chat/completions -> base_url + endpoint = .../v1/v1/chat/completions
        base_url = "http://localhost:11434/v1"
        endpoint = "/v1/chat/completions"
        url = base_url.rstrip("/") + endpoint

        request_hash = normalize_inference_request(
            "POST",
            url,
            {},
            {"model": "llama3.2:3b", "messages": [{"role": "user", "content": "Legacy error test"}]},
        )
        legacy_recording = {
            "test_id": None,
            "request": {
                "method": "POST",
                "url": url,
                "endpoint": endpoint,
                "body": {"model": "llama3.2:3b", "messages": [{"role": "user", "content": "Legacy error test"}]},
            },
            "response": {
                "body": None,
                "is_streaming": False,
                "is_exception": True,
                "exception_message": "Legacy formatted error",
            },
            "id_normalization_mapping": {},
        }
        with open(recordings_dir / f"{request_hash}.json", "w") as f:
            json.dump(legacy_recording, f, indent=2)

        with patch("openai.resources.chat.completions.AsyncCompletions.create"):
            with api_recording(mode=APIRecordingMode.REPLAY, storage_dir=str(temp_storage_dir)):
                client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="test")

                with pytest.raises(Exception) as exc_info:
                    await client.chat.completions.create(
                        model="llama3.2:3b",
                        messages=[{"role": "user", "content": "Legacy error test"}],
                    )

                assert str(exc_info.value) == "Legacy formatted error"

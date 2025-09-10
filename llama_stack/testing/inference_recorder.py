# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations  # for forward references

import hashlib
import json
import os
from collections.abc import Generator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, cast

from llama_stack.log import get_logger

logger = get_logger(__name__, category="testing")

# Global state for the recording system
_current_mode: str | None = None
_current_storage: ResponseStorage | None = None
_original_methods: dict[str, Any] = {}

from openai.types.completion_choice import CompletionChoice

# update the "finish_reason" field, since its type definition is wrong (no None is accepted)
CompletionChoice.model_fields["finish_reason"].annotation = Literal["stop", "length", "content_filter"] | None
CompletionChoice.model_rebuild()

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_STORAGE_DIR = REPO_ROOT / "tests/integration/recordings"


class InferenceMode(StrEnum):
    LIVE = "live"
    RECORD = "record"
    REPLAY = "replay"


def normalize_request(method: str, url: str, headers: dict[str, Any], body: dict[str, Any]) -> str:
    """Create a normalized hash of the request for consistent matching."""
    # Extract just the endpoint path
    from urllib.parse import urlparse

    parsed = urlparse(url)
    normalized = {"method": method.upper(), "endpoint": parsed.path, "body": body}

    # Create hash - sort_keys=True ensures deterministic ordering
    normalized_json = json.dumps(normalized, sort_keys=True)
    return hashlib.sha256(normalized_json.encode()).hexdigest()


def get_inference_mode() -> InferenceMode:
    return InferenceMode(os.environ.get("LLAMA_STACK_TEST_INFERENCE_MODE", "replay").lower())


def setup_inference_recording():
    """
    Returns a context manager that can be used to record or replay inference requests. This is to be used in tests
    to increase their reliability and reduce reliance on expensive, external services.

    Currently, this is only supported for OpenAI and Ollama clients. These should cover the vast majority of use cases.

    Two environment variables are supported:
    - LLAMA_STACK_TEST_INFERENCE_MODE: The mode to run in. Must be 'live', 'record', or 'replay'. Default is 'replay'.
    - LLAMA_STACK_TEST_RECORDING_DIR: The directory to store the recordings in. Default is 'tests/integration/recordings'.

    The recordings are stored as JSON files.
    """
    mode = get_inference_mode()
    if mode == InferenceMode.LIVE:
        return None

    storage_dir = os.environ.get("LLAMA_STACK_TEST_RECORDING_DIR", DEFAULT_STORAGE_DIR)
    return inference_recording(mode=mode, storage_dir=storage_dir)


def _serialize_response(response: Any) -> Any:
    if hasattr(response, "model_dump"):
        data = response.model_dump(mode="json")
        return {
            "__type__": f"{response.__class__.__module__}.{response.__class__.__qualname__}",
            "__data__": data,
        }
    elif hasattr(response, "__dict__"):
        return dict(response.__dict__)
    else:
        return response


def _deserialize_response(data: dict[str, Any]) -> Any:
    # Check if this is a serialized Pydantic model with type information
    if isinstance(data, dict) and "__type__" in data and "__data__" in data:
        try:
            # Import the original class and reconstruct the object
            module_path, class_name = data["__type__"].rsplit(".", 1)
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)

            if not hasattr(cls, "model_validate"):
                raise ValueError(f"Pydantic class {cls} does not support model_validate?")

            return cls.model_validate(data["__data__"])
        except (ImportError, AttributeError, TypeError, ValueError) as e:
            logger.warning(f"Failed to deserialize object of type {data['__type__']}: {e}")
            return data["__data__"]

    return data


class ResponseStorage:
    """Handles SQLite index + JSON file storage/retrieval for inference recordings."""

    def __init__(self, test_dir: Path):
        self.test_dir = test_dir
        self.responses_dir = self.test_dir / "responses"

        self._ensure_directories()

    def _ensure_directories(self):
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.responses_dir.mkdir(exist_ok=True)

    def store_recording(self, request_hash: str, request: dict[str, Any], response: dict[str, Any]):
        """Store a request/response pair."""
        # Generate unique response filename
        short_hash = request_hash[:12]
        response_file = f"{short_hash}.json"

        # Serialize response body if needed
        serialized_response = dict(response)
        if "body" in serialized_response:
            if isinstance(serialized_response["body"], list):
                # Handle streaming responses (list of chunks)
                serialized_response["body"] = [_serialize_response(chunk) for chunk in serialized_response["body"]]
            else:
                # Handle single response
                serialized_response["body"] = _serialize_response(serialized_response["body"])

        # If this is an Ollama /api/tags recording, include models digest in filename to distinguish variants
        endpoint = request.get("endpoint")
        if endpoint in ("/api/tags", "/v1/models"):
            digest = _model_identifiers_digest(endpoint, response)
            response_file = f"models-{short_hash}-{digest}.json"

        response_path = self.responses_dir / response_file

        # Save response to JSON file
        with open(response_path, "w") as f:
            json.dump({"request": request, "response": serialized_response}, f, indent=2)
            f.write("\n")
            f.flush()

    def find_recording(self, request_hash: str) -> dict[str, Any] | None:
        """Find a recorded response by request hash."""
        response_file = f"{request_hash[:12]}.json"
        response_path = self.responses_dir / response_file

        if not response_path.exists():
            return None

        return _recording_from_file(response_path)

    def _model_list_responses(self, short_hash: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for path in self.responses_dir.glob(f"models-{short_hash}-*.json"):
            data = _recording_from_file(path)
            results.append(data)
        return results


def _recording_from_file(response_path) -> dict[str, Any]:
    with open(response_path) as f:
        data = json.load(f)

    # Deserialize response body if needed
    if "response" in data and "body" in data["response"]:
        if isinstance(data["response"]["body"], list):
            # Handle streaming responses
            data["response"]["body"] = [_deserialize_response(chunk) for chunk in data["response"]["body"]]
        else:
            # Handle single response
            data["response"]["body"] = _deserialize_response(data["response"]["body"])

    return cast(dict[str, Any], data)


def _model_identifiers_digest(endpoint: str, response: dict[str, Any]) -> str:
    def _extract_model_identifiers():
        """Extract a stable set of identifiers for model-list endpoints.

        Supported endpoints:
        - '/api/tags' (Ollama): response body has 'models': [ { name/model/digest/id/... }, ... ]
        - '/v1/models' (OpenAI): response body has 'data': [ { id: ... }, ... ]
        Returns a list of unique identifiers or None if structure doesn't match.
        """
        body = response["body"]
        if endpoint == "/api/tags":
            items = body.get("models")
            idents = [m.model for m in items]
        else:
            items = body.get("data")
            idents = [m.id for m in items]
        return sorted(set(idents))

    identifiers = _extract_model_identifiers()
    return hashlib.sha1(("|".join(identifiers)).encode("utf-8")).hexdigest()[:8]


def _combine_model_list_responses(endpoint: str, records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return a single, unioned recording for supported model-list endpoints."""
    seen: dict[str, dict[str, Any]] = {}
    for rec in records:
        body = rec["response"]["body"]
        if endpoint == "/api/tags":
            items = body.models
        elif endpoint == "/v1/models":
            items = body.data
        else:
            items = []

        for m in items:
            if endpoint == "/v1/models":
                key = m.id
            else:
                key = m.model
            seen[key] = m

    ordered = [seen[k] for k in sorted(seen.keys())]
    canonical = records[0]
    canonical_req = canonical.get("request", {})
    if isinstance(canonical_req, dict):
        canonical_req["endpoint"] = endpoint
    if endpoint == "/v1/models":
        body = {"data": ordered, "object": "list"}
    else:
        from ollama import ListResponse

        body = ListResponse(models=ordered)
    return {"request": canonical_req, "response": {"body": body, "is_streaming": False}}


async def _patched_inference_method(original_method, self, client_type, endpoint, *args, **kwargs):
    global _current_mode, _current_storage

    if _current_mode == InferenceMode.LIVE or _current_storage is None:
        # Normal operation
        return await original_method(self, *args, **kwargs)

    # Get base URL based on client type
    if client_type == "openai":
        base_url = str(self._client.base_url)
    elif client_type == "ollama":
        # Get base URL from the client (Ollama client uses host attribute)
        base_url = getattr(self, "host", "http://localhost:11434")
        if not base_url.startswith("http"):
            base_url = f"http://{base_url}"
    else:
        raise ValueError(f"Unknown client type: {client_type}")

    url = base_url.rstrip("/") + endpoint
    method = "POST"
    headers = {}
    body = kwargs

    request_hash = normalize_request(method, url, headers, body)

    if _current_mode == InferenceMode.REPLAY:
        # Special handling for model-list endpoints: return union of all responses
        if endpoint in ("/api/tags", "/v1/models"):
            records = _current_storage._model_list_responses(request_hash[:12])
            recording = _combine_model_list_responses(endpoint, records)
        else:
            recording = _current_storage.find_recording(request_hash)
        if recording:
            response_body = recording["response"]["body"]

            if recording["response"].get("is_streaming", False):

                async def replay_stream():
                    for chunk in response_body:
                        yield chunk

                return replay_stream()
            else:
                return response_body
        else:
            raise RuntimeError(
                f"No recorded response found for request hash: {request_hash}\n"
                f"Request: {method} {url} {body}\n"
                f"Model: {body.get('model', 'unknown')}\n"
                f"To record this response, run with LLAMA_STACK_TEST_INFERENCE_MODE=record"
            )

    elif _current_mode == InferenceMode.RECORD:
        response = await original_method(self, *args, **kwargs)

        request_data = {
            "method": method,
            "url": url,
            "headers": headers,
            "body": body,
            "endpoint": endpoint,
            "model": body.get("model", ""),
        }

        # Determine if this is a streaming request based on request parameters
        is_streaming = body.get("stream", False)

        if is_streaming:
            # For streaming responses, we need to collect all chunks immediately before yielding
            # This ensures the recording is saved even if the generator isn't fully consumed
            chunks = []
            async for chunk in response:
                chunks.append(chunk)

            # Store the recording immediately
            response_data = {"body": chunks, "is_streaming": True}
            _current_storage.store_recording(request_hash, request_data, response_data)

            # Return a generator that replays the stored chunks
            async def replay_recorded_stream():
                for chunk in chunks:
                    yield chunk

            return replay_recorded_stream()
        else:
            response_data = {"body": response, "is_streaming": False}
            _current_storage.store_recording(request_hash, request_data, response_data)
            return response

    else:
        raise AssertionError(f"Invalid mode: {_current_mode}")


def patch_inference_clients():
    """Install monkey patches for OpenAI client methods and Ollama AsyncClient methods."""
    global _original_methods

    from ollama import AsyncClient as OllamaAsyncClient
    from openai.resources.chat.completions import AsyncCompletions as AsyncChatCompletions
    from openai.resources.completions import AsyncCompletions
    from openai.resources.embeddings import AsyncEmbeddings
    from openai.resources.models import AsyncModels

    # Store original methods for both OpenAI and Ollama clients
    _original_methods = {
        "chat_completions_create": AsyncChatCompletions.create,
        "completions_create": AsyncCompletions.create,
        "embeddings_create": AsyncEmbeddings.create,
        "models_list": AsyncModels.list,
        "ollama_generate": OllamaAsyncClient.generate,
        "ollama_chat": OllamaAsyncClient.chat,
        "ollama_embed": OllamaAsyncClient.embed,
        "ollama_ps": OllamaAsyncClient.ps,
        "ollama_pull": OllamaAsyncClient.pull,
        "ollama_list": OllamaAsyncClient.list,
    }

    # Create patched methods for OpenAI client
    async def patched_chat_completions_create(self, *args, **kwargs):
        return await _patched_inference_method(
            _original_methods["chat_completions_create"], self, "openai", "/v1/chat/completions", *args, **kwargs
        )

    async def patched_completions_create(self, *args, **kwargs):
        return await _patched_inference_method(
            _original_methods["completions_create"], self, "openai", "/v1/completions", *args, **kwargs
        )

    async def patched_embeddings_create(self, *args, **kwargs):
        return await _patched_inference_method(
            _original_methods["embeddings_create"], self, "openai", "/v1/embeddings", *args, **kwargs
        )

    async def patched_models_list(self, *args, **kwargs):
        return await _patched_inference_method(
            _original_methods["models_list"], self, "openai", "/v1/models", *args, **kwargs
        )

    # Apply OpenAI patches
    AsyncChatCompletions.create = patched_chat_completions_create
    AsyncCompletions.create = patched_completions_create
    AsyncEmbeddings.create = patched_embeddings_create
    AsyncModels.list = patched_models_list

    # Create patched methods for Ollama client
    async def patched_ollama_generate(self, *args, **kwargs):
        return await _patched_inference_method(
            _original_methods["ollama_generate"], self, "ollama", "/api/generate", *args, **kwargs
        )

    async def patched_ollama_chat(self, *args, **kwargs):
        return await _patched_inference_method(
            _original_methods["ollama_chat"], self, "ollama", "/api/chat", *args, **kwargs
        )

    async def patched_ollama_embed(self, *args, **kwargs):
        return await _patched_inference_method(
            _original_methods["ollama_embed"], self, "ollama", "/api/embeddings", *args, **kwargs
        )

    async def patched_ollama_ps(self, *args, **kwargs):
        return await _patched_inference_method(
            _original_methods["ollama_ps"], self, "ollama", "/api/ps", *args, **kwargs
        )

    async def patched_ollama_pull(self, *args, **kwargs):
        return await _patched_inference_method(
            _original_methods["ollama_pull"], self, "ollama", "/api/pull", *args, **kwargs
        )

    async def patched_ollama_list(self, *args, **kwargs):
        return await _patched_inference_method(
            _original_methods["ollama_list"], self, "ollama", "/api/tags", *args, **kwargs
        )

    # Apply Ollama patches
    OllamaAsyncClient.generate = patched_ollama_generate
    OllamaAsyncClient.chat = patched_ollama_chat
    OllamaAsyncClient.embed = patched_ollama_embed
    OllamaAsyncClient.ps = patched_ollama_ps
    OllamaAsyncClient.pull = patched_ollama_pull
    OllamaAsyncClient.list = patched_ollama_list


def unpatch_inference_clients():
    """Remove monkey patches and restore original OpenAI and Ollama client methods."""
    global _original_methods

    if not _original_methods:
        return

    # Import here to avoid circular imports
    from ollama import AsyncClient as OllamaAsyncClient
    from openai.resources.chat.completions import AsyncCompletions as AsyncChatCompletions
    from openai.resources.completions import AsyncCompletions
    from openai.resources.embeddings import AsyncEmbeddings
    from openai.resources.models import AsyncModels

    # Restore OpenAI client methods
    AsyncChatCompletions.create = _original_methods["chat_completions_create"]
    AsyncCompletions.create = _original_methods["completions_create"]
    AsyncEmbeddings.create = _original_methods["embeddings_create"]
    AsyncModels.list = _original_methods["models_list"]

    # Restore Ollama client methods if they were patched
    OllamaAsyncClient.generate = _original_methods["ollama_generate"]
    OllamaAsyncClient.chat = _original_methods["ollama_chat"]
    OllamaAsyncClient.embed = _original_methods["ollama_embed"]
    OllamaAsyncClient.ps = _original_methods["ollama_ps"]
    OllamaAsyncClient.pull = _original_methods["ollama_pull"]
    OllamaAsyncClient.list = _original_methods["ollama_list"]

    _original_methods.clear()


@contextmanager
def inference_recording(mode: str, storage_dir: str | Path | None = None) -> Generator[None, None, None]:
    """Context manager for inference recording/replaying."""
    global _current_mode, _current_storage

    # Store previous state
    prev_mode = _current_mode
    prev_storage = _current_storage

    try:
        _current_mode = mode

        if mode in ["record", "replay"]:
            if storage_dir is None:
                raise ValueError("storage_dir is required for record and replay modes")
            _current_storage = ResponseStorage(Path(storage_dir))
            patch_inference_clients()

        yield

    finally:
        # Restore previous state
        if mode in ["record", "replay"]:
            unpatch_inference_clients()

        _current_mode = prev_mode
        _current_storage = prev_storage

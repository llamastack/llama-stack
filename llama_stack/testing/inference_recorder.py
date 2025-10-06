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

from openai import NOT_GIVEN

from llama_stack.log import get_logger

logger = get_logger(__name__, category="testing")

# Global state for the recording system
# Note: Using module globals instead of ContextVars because the session-scoped
# client initialization happens in one async context, but tests run in different
# contexts, and we need the mode/storage to persist across all contexts.
_current_mode: str | None = None
_current_storage: ResponseStorage | None = None
_original_methods: dict[str, Any] = {}

# Test context uses ContextVar since it changes per-test and needs async isolation
from contextvars import ContextVar

_test_context: ContextVar[str | None] = ContextVar("_test_context", default=None)

from openai.types.completion_choice import CompletionChoice

# update the "finish_reason" field, since its type definition is wrong (no None is accepted)
CompletionChoice.model_fields["finish_reason"].annotation = Literal["stop", "length", "content_filter"] | None
CompletionChoice.model_rebuild()

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_STORAGE_DIR = REPO_ROOT / "tests/integration/common"


class InferenceMode(StrEnum):
    LIVE = "live"
    RECORD = "record"
    REPLAY = "replay"
    RECORD_IF_MISSING = "record-if-missing"


def normalize_request(method: str, url: str, headers: dict[str, Any], body: dict[str, Any]) -> str:
    """Create a normalized hash of the request for consistent matching.

    Includes test_id from context to ensure test isolation - identical requests
    from different tests will have different hashes.

    Exception: Model list endpoints (/v1/models, /api/tags) exclude test_id since
    they are infrastructure/shared and need to work across session setup and tests.
    """
    # Extract just the endpoint path
    from urllib.parse import urlparse

    parsed = urlparse(url)
    normalized: dict[str, Any] = {
        "method": method.upper(),
        "endpoint": parsed.path,
        "body": body,
    }

    # Include test_id for isolation, except for shared infrastructure endpoints
    if parsed.path not in ("/api/tags", "/v1/models"):
        normalized["test_id"] = _test_context.get()

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
    - LLAMA_STACK_TEST_INFERENCE_MODE: The mode to run in. Must be 'live', 'record', 'replay', or 'record-if-missing'. Default is 'replay'.
      - 'live': Make all requests live without recording
      - 'record': Record all requests (overwrites existing recordings)
      - 'replay': Use only recorded responses (fails if recording not found)
      - 'record-if-missing': Use recorded responses when available, record new ones when not found
    - LLAMA_STACK_TEST_RECORDING_DIR: The directory to store the recordings in. Default is 'tests/integration/recordings'.

    The recordings are stored as JSON files.
    """
    mode = get_inference_mode()
    if mode == InferenceMode.LIVE:
        return None

    storage_dir = os.environ.get("LLAMA_STACK_TEST_RECORDING_DIR", DEFAULT_STORAGE_DIR)
    return inference_recording(mode=mode, storage_dir=storage_dir)


def _normalize_response_data(data: dict[str, Any], request_hash: str) -> dict[str, Any]:
    """Normalize fields that change between recordings but don't affect functionality.

    This reduces noise in git diffs by making IDs deterministic and timestamps constant.
    """
    # Only normalize ID for completion/chat responses, not for model objects
    # Model objects have "object": "model" and the ID is the actual model identifier
    if "id" in data and data.get("object") != "model":
        data["id"] = f"rec-{request_hash[:12]}"

    # Normalize timestamp to epoch (0) (for OpenAI-style responses)
    # But not for model objects where created timestamp might be meaningful
    if "created" in data and data.get("object") != "model":
        data["created"] = 0

    # Normalize Ollama-specific timestamp fields
    if "created_at" in data:
        data["created_at"] = "1970-01-01T00:00:00.000000Z"

    # Normalize Ollama-specific duration fields (these vary based on system load)
    if "total_duration" in data and data["total_duration"] is not None:
        data["total_duration"] = 0
    if "load_duration" in data and data["load_duration"] is not None:
        data["load_duration"] = 0
    if "prompt_eval_duration" in data and data["prompt_eval_duration"] is not None:
        data["prompt_eval_duration"] = 0
    if "eval_duration" in data and data["eval_duration"] is not None:
        data["eval_duration"] = 0

    return data


def _serialize_response(response: Any, request_hash: str = "") -> Any:
    if hasattr(response, "model_dump"):
        data = response.model_dump(mode="json")
        # Normalize fields to reduce noise
        data = _normalize_response_data(data, request_hash)
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
            logger.warning(f"Failed to deserialize object of type {data['__type__']} with model_validate: {e}")
            try:
                return cls.model_construct(**data["__data__"])
            except Exception as e:
                logger.warning(f"Failed to deserialize object of type {data['__type__']} with model_construct: {e}")
                return data["__data__"]

    return data


class ResponseStorage:
    """Handles SQLite index + JSON file storage/retrieval for inference recordings."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        # Don't create responses_dir here - determine it per-test at runtime

    def _get_test_dir(self) -> Path:
        """Get the recordings directory in the test file's parent directory.

        For test at "tests/integration/inference/test_foo.py::test_bar",
        returns "tests/integration/inference/recordings/".
        """
        test_id = _test_context.get()
        if test_id:
            # Extract the directory path from the test nodeid
            # e.g., "tests/integration/inference/test_basic.py::test_foo[params]"
            # -> get "tests/integration/inference"
            test_file = test_id.split("::")[0]  # Remove test function part
            test_dir = Path(test_file).parent  # Get parent directory

            # Put recordings in a "recordings" subdirectory of the test's parent dir
            # e.g., "tests/integration/inference" -> "tests/integration/inference/recordings"
            return test_dir / "recordings"
        else:
            # Fallback for non-test contexts
            return self.base_dir / "recordings"

    def _ensure_directories(self):
        """Ensure test-specific directories exist."""
        test_dir = self._get_test_dir()
        test_dir.mkdir(parents=True, exist_ok=True)
        return test_dir

    def store_recording(self, request_hash: str, request: dict[str, Any], response: dict[str, Any]):
        """Store a request/response pair."""
        responses_dir = self._ensure_directories()

        # Use FULL hash (not truncated)
        response_file = f"{request_hash}.json"

        # Serialize response body if needed
        serialized_response = dict(response)
        if "body" in serialized_response:
            if isinstance(serialized_response["body"], list):
                # Handle streaming responses (list of chunks)
                serialized_response["body"] = [
                    _serialize_response(chunk, request_hash) for chunk in serialized_response["body"]
                ]
            else:
                # Handle single response
                serialized_response["body"] = _serialize_response(serialized_response["body"], request_hash)

        # For model-list endpoints, include digest in filename to distinguish different model sets
        endpoint = request.get("endpoint")
        if endpoint in ("/api/tags", "/v1/models"):
            digest = _model_identifiers_digest(endpoint, response)
            response_file = f"models-{request_hash}-{digest}.json"

        response_path = responses_dir / response_file

        # Save response to JSON file with metadata
        with open(response_path, "w") as f:
            json.dump(
                {
                    "test_id": _test_context.get(),  # Include for debugging
                    "request": request,
                    "response": serialized_response,
                },
                f,
                indent=2,
            )
            f.write("\n")
            f.flush()

    def find_recording(self, request_hash: str) -> dict[str, Any] | None:
        """Find a recorded response by request hash.

        Uses fallback: first checks test-specific dir, then falls back to base recordings dir.
        This handles cases where recordings happen during session setup (no test context) but
        are requested during tests (with test context).
        """
        response_file = f"{request_hash}.json"

        # Try test-specific directory first
        test_dir = self._get_test_dir()
        response_path = test_dir / response_file

        if response_path.exists():
            return _recording_from_file(response_path)

        # Fallback to base recordings directory (for session-level recordings)
        fallback_dir = self.base_dir / "recordings"
        fallback_path = fallback_dir / response_file

        if fallback_path.exists():
            return _recording_from_file(fallback_path)

        return None

    def _model_list_responses(self, request_hash: str) -> list[dict[str, Any]]:
        """Find all model-list recordings with the given hash (different digests)."""
        results: list[dict[str, Any]] = []

        # Check test-specific directory first
        test_dir = self._get_test_dir()
        if test_dir.exists():
            for path in test_dir.glob(f"models-{request_hash}-*.json"):
                data = _recording_from_file(path)
                results.append(data)

        # Also check fallback directory
        fallback_dir = self.base_dir / "recordings"
        if fallback_dir.exists():
            for path in fallback_dir.glob(f"models-{request_hash}-*.json"):
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
    """Generate a digest from model identifiers for distinguishing different model sets."""

    def _extract_model_identifiers():
        """Extract a stable set of identifiers for model-list endpoints.

        Supported endpoints:
        - '/api/tags' (Ollama): response body has 'models': [ { name/model/digest/id/... }, ... ]
        - '/v1/models' (OpenAI): response body is: [ { id: ... }, ... ]
        Returns a list of unique identifiers or None if structure doesn't match.
        """
        if "models" in response["body"]:
            # ollama
            items = response["body"]["models"]
        else:
            # openai
            items = response["body"]
        idents = [m.model if endpoint == "/api/tags" else m.id for m in items]
        return sorted(set(idents))

    identifiers = _extract_model_identifiers()
    return hashlib.sha256(("|".join(identifiers)).encode("utf-8")).hexdigest()[:8]


def _combine_model_list_responses(endpoint: str, records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return a single, unioned recording for supported model-list endpoints.

    Merges multiple recordings with different model sets (from different servers) into
    a single response containing all models.
    """
    if not records:
        return None

    seen: dict[str, dict[str, Any]] = {}
    for rec in records:
        body = rec["response"]["body"]
        if endpoint == "/v1/models":
            for m in body:
                key = m.id
                seen[key] = m
        elif endpoint == "/api/tags":
            for m in body.models:
                key = m.model
                seen[key] = m

    ordered = [seen[k] for k in sorted(seen.keys())]
    canonical = records[0]
    canonical_req = canonical.get("request", {})
    if isinstance(canonical_req, dict):
        canonical_req["endpoint"] = endpoint
    body = ordered
    if endpoint == "/api/tags":
        from ollama import ListResponse

        body = ListResponse(models=ordered)
    return {"request": canonical_req, "response": {"body": body, "is_streaming": False}}


async def _patched_inference_method(original_method, self, client_type, endpoint, *args, **kwargs):
    global _current_mode, _current_storage

    mode = _current_mode
    storage = _current_storage

    if mode == InferenceMode.LIVE or storage is None:
        if endpoint == "/v1/models":
            return original_method(self, *args, **kwargs)
        else:
            return await original_method(self, *args, **kwargs)

    # Get base URL based on client type
    if client_type == "openai":
        base_url = str(self._client.base_url)

        # the OpenAI client methods may pass NOT_GIVEN for unset parameters; filter these out
        kwargs = {k: v for k, v in kwargs.items() if v is not NOT_GIVEN}
    elif client_type == "ollama":
        # Get base URL from the client (Ollama client uses host attribute)
        base_url = getattr(self, "host", "http://localhost:11434")
        if not base_url.startswith("http"):
            base_url = f"http://{base_url}"
    else:
        raise ValueError(f"Unknown client type: {client_type}")

    url = base_url.rstrip("/") + endpoint
    # Special handling for Databricks URLs to avoid leaking workspace info
    # e.g. https://adb-1234567890123456.7.cloud.databricks.com -> https://...cloud.databricks.com
    if "cloud.databricks.com" in url:
        url = "__databricks__" + url.split("cloud.databricks.com")[-1]
    method = "POST"
    headers = {}
    body = kwargs

    request_hash = normalize_request(method, url, headers, body)

    # Try to find existing recording for REPLAY or RECORD_IF_MISSING modes
    recording = None
    if mode == InferenceMode.REPLAY or mode == InferenceMode.RECORD_IF_MISSING:
        # Special handling for model-list endpoints: merge all recordings with this hash
        if endpoint in ("/api/tags", "/v1/models"):
            records = storage._model_list_responses(request_hash)
            recording = _combine_model_list_responses(endpoint, records)
        else:
            recording = storage.find_recording(request_hash)

        if recording:
            response_body = recording["response"]["body"]

            if recording["response"].get("is_streaming", False):

                async def replay_stream():
                    for chunk in response_body:
                        yield chunk

                return replay_stream()
            else:
                return response_body
        elif mode == InferenceMode.REPLAY:
            # REPLAY mode requires recording to exist
            raise RuntimeError(
                f"No recorded response found for request hash: {request_hash}\n"
                f"Request: {method} {url} {body}\n"
                f"Model: {body.get('model', 'unknown')}\n"
                f"To record this response, run with LLAMA_STACK_TEST_INFERENCE_MODE=record"
            )

    if mode == InferenceMode.RECORD or (mode == InferenceMode.RECORD_IF_MISSING and not recording):
        if endpoint == "/v1/models":
            response = original_method(self, *args, **kwargs)
        else:
            response = await original_method(self, *args, **kwargs)

        # we want to store the result of the iterator, not the iterator itself
        if endpoint == "/v1/models":
            response = [m async for m in response]

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
            storage.store_recording(request_hash, request_data, response_data)

            # Return a generator that replays the stored chunks
            async def replay_recorded_stream():
                for chunk in chunks:
                    yield chunk

            return replay_recorded_stream()
        else:
            response_data = {"body": response, "is_streaming": False}
            storage.store_recording(request_hash, request_data, response_data)
            return response

    else:
        raise AssertionError(f"Invalid mode: {mode}")


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

    def patched_models_list(self, *args, **kwargs):
        async def _iter():
            for item in await _patched_inference_method(
                _original_methods["models_list"], self, "openai", "/v1/models", *args, **kwargs
            ):
                yield item

        return _iter()

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

        if mode in ["record", "replay", "record-if-missing"]:
            if storage_dir is None:
                raise ValueError("storage_dir is required for record, replay, and record-if-missing modes")
            _current_storage = ResponseStorage(Path(storage_dir))
            patch_inference_clients()

        yield

    finally:
        # Restore previous state
        if mode in ["record", "replay", "record-if-missing"]:
            unpatch_inference_clients()

        _current_mode = prev_mode
        _current_storage = prev_storage

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations  # for forward references

import hashlib
import json
import os
import re
from collections.abc import Callable, Generator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, cast

from openai import NOT_GIVEN, OpenAI

from llama_stack.core.id_generation import reset_id_override, set_id_override
from llama_stack.log import get_logger

logger = get_logger(__name__, category="testing")

# Global state for the recording system
# Note: Using module globals instead of ContextVars because the session-scoped
# client initialization happens in one async context, but tests run in different
# contexts, and we need the mode/storage to persist across all contexts.
_current_mode: str | None = None
_current_storage: ResponseStorage | None = None
_original_methods: dict[str, Any] = {}

# Per-test deterministic ID counters (test_id -> id_kind -> counter)
_id_counters: dict[str, dict[str, int]] = {}

# Test context uses ContextVar since it changes per-test and needs async isolation
from openai.types.completion_choice import CompletionChoice

from llama_stack.core.testing_context import get_test_context, is_debug_mode

# update the "finish_reason" field, since its type definition is wrong (no None is accepted)
CompletionChoice.model_fields["finish_reason"].annotation = cast(
    type[Any] | None, Literal["stop", "length", "content_filter"] | None
)
CompletionChoice.model_rebuild()

REPO_ROOT = Path(__file__).parent.parent.parent.parent
DEFAULT_STORAGE_DIR = REPO_ROOT / "tests/integration/common"


class APIRecordingMode(StrEnum):
    LIVE = "live"
    RECORD = "record"
    REPLAY = "replay"
    RECORD_IF_MISSING = "record-if-missing"


_ID_KIND_PREFIXES: dict[str, str] = {
    "file": "file-",
    "vector_store": "vs_",
    "vector_store_file_batch": "batch_",
    "tool_call": "call_",
}


_FLOAT_IN_STRING_PATTERN = re.compile(r"(-?\d+\.\d{4,})")


def _normalize_numeric_literal_strings(value: str) -> str:
    """Round any long decimal literals embedded in strings for stable hashing."""

    def _replace(match: re.Match[str]) -> str:
        number = float(match.group(0))
        return f"{number:.5f}"

    return _FLOAT_IN_STRING_PATTERN.sub(_replace, value)


def _normalize_body_for_hash(value: Any) -> Any:
    """Recursively normalize a JSON-like value to improve hash stability."""

    if isinstance(value, dict):
        return {key: _normalize_body_for_hash(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_body_for_hash(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_body_for_hash(item) for item in value)
    if isinstance(value, float):
        return round(value, 5)
    if isinstance(value, str):
        return _normalize_numeric_literal_strings(value)
    return value


def _allocate_test_scoped_id(kind: str) -> str | None:
    """Return the next deterministic ID for the given kind within the current test."""

    global _id_counters

    test_id = get_test_context()
    prefix = _ID_KIND_PREFIXES.get(kind)

    if prefix is None:
        return None

    if not test_id:
        raise ValueError(f"Test ID is required for {kind} ID allocation")

    key = test_id
    if key not in _id_counters:
        _id_counters[key] = {}

    # each test should get a contiguous block of IDs otherwise we will get
    # collisions between tests inside other systems (like file storage) which
    # expect IDs to be unique
    test_hash = hashlib.sha256(test_id.encode()).hexdigest()
    test_hash_int = int(test_hash, 16)
    counter = test_hash_int % 1000000000000

    counter = _id_counters[key].get(kind, counter) + 1
    _id_counters[key][kind] = counter

    return f"{prefix}{counter}"


def _deterministic_id_override(kind: str, factory: Callable[[], str]) -> str:
    deterministic_id = _allocate_test_scoped_id(kind)
    if deterministic_id is not None:
        return deterministic_id
    return factory()


def normalize_inference_request(method: str, url: str, headers: dict[str, Any], body: dict[str, Any]) -> str:
    """Create a normalized hash of the request for consistent matching.

    Includes test_id from context to ensure test isolation - identical requests
    from different tests will have different hashes.

    Exception: Model list endpoints (/v1/models, /api/tags) exclude test_id since
    they are infrastructure/shared and need to work across session setup and tests.
    """

    # Extract just the endpoint path
    from urllib.parse import urlparse

    parsed = urlparse(url)

    body_for_hash = _normalize_body_for_hash(body)

    test_id = get_test_context()
    normalized: dict[str, Any] = {
        "method": method.upper(),
        "endpoint": parsed.path,
        "body": body_for_hash,
    }

    # Include test_id for isolation, except for shared infrastructure endpoints
    if parsed.path not in ("/api/tags", "/v1/models", "/v1/openai/v1/models"):
        normalized["test_id"] = test_id

    normalized_json = json.dumps(normalized, sort_keys=True)
    request_hash = hashlib.sha256(normalized_json.encode()).hexdigest()

    if is_debug_mode():
        logger.info("[RECORDING DEBUG] Hash computation:")
        logger.info(f"  Test ID: {test_id}")
        logger.info(f"  Method: {method.upper()}")
        logger.info(f"  Endpoint: {parsed.path}")
        logger.info(f"  Model: {body.get('model', 'N/A')}")
        logger.info(f"  Computed hash: {request_hash}")

    return request_hash


def normalize_tool_request(provider_name: str, tool_name: str, kwargs: dict[str, Any]) -> str:
    """Create a normalized hash of the tool request for consistent matching."""
    normalized = {
        "provider": provider_name,
        "tool_name": tool_name,
        "kwargs": kwargs,
    }

    # Create hash - sort_keys=True ensures deterministic ordering
    normalized_json = json.dumps(normalized, sort_keys=True)
    return hashlib.sha256(normalized_json.encode()).hexdigest()


def patch_httpx_for_test_id():
    """Patch client _prepare_request methods to inject test ID into provider data header.

    This is needed for server mode where the test ID must be transported from
    client to server via HTTP headers. In library_client mode, this patch is a no-op
    since everything runs in the same process.

    We use the _prepare_request hook that Stainless clients provide for mutating
    requests after construction but before sending.
    """
    from llama_stack_client import LlamaStackClient

    if "llama_stack_client_prepare_request" in _original_methods:
        return

    _original_methods["llama_stack_client_prepare_request"] = LlamaStackClient._prepare_request
    _original_methods["openai_prepare_request"] = OpenAI._prepare_request

    def patched_prepare_request(self, request):
        # Call original first (it's a sync method that returns None)
        # Determine which original to call based on client type
        _original_methods["llama_stack_client_prepare_request"](self, request)
        _original_methods["openai_prepare_request"](self, request)

        # Only inject test ID in server mode
        stack_config_type = os.environ.get("LLAMA_STACK_TEST_STACK_CONFIG_TYPE", "library_client")
        test_id = get_test_context()

        if stack_config_type == "server" and test_id:
            provider_data_header = request.headers.get("X-LlamaStack-Provider-Data")

            if provider_data_header:
                provider_data = json.loads(provider_data_header)
            else:
                provider_data = {}

            provider_data["__test_id"] = test_id
            request.headers["X-LlamaStack-Provider-Data"] = json.dumps(provider_data)

            if is_debug_mode():
                logger.info("[RECORDING DEBUG] Injected test ID into request header:")
                logger.info(f"  Test ID: {test_id}")
                logger.info(f"  URL: {request.url}")

        return None

    LlamaStackClient._prepare_request = patched_prepare_request
    OpenAI._prepare_request = patched_prepare_request


# currently, unpatch is never called
def unpatch_httpx_for_test_id():
    """Remove client _prepare_request patches for test ID injection."""
    if "llama_stack_client_prepare_request" not in _original_methods:
        return

    from llama_stack_client import LlamaStackClient

    LlamaStackClient._prepare_request = _original_methods["llama_stack_client_prepare_request"]
    del _original_methods["llama_stack_client_prepare_request"]
    OpenAI._prepare_request = _original_methods["openai_prepare_request"]
    del _original_methods["openai_prepare_request"]


def get_api_recording_mode() -> APIRecordingMode:
    return APIRecordingMode(os.environ.get("LLAMA_STACK_TEST_INFERENCE_MODE", "replay").lower())


def setup_api_recording():
    """
    Returns a context manager that can be used to record or replay API requests (inference and tools).
    This is to be used in tests to increase their reliability and reduce reliance on expensive, external services.

    Currently supports:
    - Inference: OpenAI and Ollama clients
    - Tools: Search providers (Tavily)

    Two environment variables are supported:
    - LLAMA_STACK_TEST_INFERENCE_MODE: The mode to run in. Must be 'live', 'record', 'replay', or 'record-if-missing'. Default is 'replay'.
      - 'live': Make all requests live without recording
      - 'record': Record all requests (overwrites existing recordings)
      - 'replay': Use only recorded responses (fails if recording not found)
      - 'record-if-missing': Use recorded responses when available, record new ones when not found
    - LLAMA_STACK_TEST_RECORDING_DIR: The directory to store the recordings in. Default is 'tests/integration/recordings'.

    The recordings are stored as JSON files.
    """
    mode = get_api_recording_mode()
    if mode == APIRecordingMode.LIVE:
        return None

    storage_dir = os.environ.get("LLAMA_STACK_TEST_RECORDING_DIR", DEFAULT_STORAGE_DIR)
    return api_recording(mode=mode, storage_dir=storage_dir)


def _normalize_response(data: dict[str, Any], request_hash: str) -> dict[str, Any]:
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
        data = _normalize_response(data, request_hash)
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
        test_id = get_test_context()
        if test_id:
            # Extract the directory path from the test nodeid
            # e.g., "tests/integration/inference/test_basic.py::test_foo[params]"
            # -> get "tests/integration/inference"
            test_file = test_id.split("::")[0]  # Remove test function part
            test_dir = Path(test_file).parent  # Get parent directory

            if self.base_dir.is_absolute():
                repo_root = self.base_dir.parent.parent.parent
                result = repo_root / test_dir / "recordings"
                if is_debug_mode():
                    logger.info("[RECORDING DEBUG] Path resolution (absolute base_dir):")
                    logger.info(f"  Test ID: {test_id}")
                    logger.info(f"  Base dir: {self.base_dir}")
                    logger.info(f"  Repo root: {repo_root}")
                    logger.info(f"  Test file: {test_file}")
                    logger.info(f"  Test dir: {test_dir}")
                    logger.info(f"  Recordings dir: {result}")
                return result
            else:
                result = test_dir / "recordings"
                if is_debug_mode():
                    logger.info("[RECORDING DEBUG] Path resolution (relative base_dir):")
                    logger.info(f"  Test ID: {test_id}")
                    logger.info(f"  Base dir: {self.base_dir}")
                    logger.info(f"  Test dir: {test_dir}")
                    logger.info(f"  Recordings dir: {result}")
                return result
        else:
            # Fallback for non-test contexts
            result = self.base_dir / "recordings"
            if is_debug_mode():
                logger.info("[RECORDING DEBUG] Path resolution (no test context):")
                logger.info(f"  Base dir: {self.base_dir}")
                logger.info(f"  Recordings dir: {result}")
            return result

    def _ensure_directory(self):
        """Ensure test-specific directories exist."""
        test_dir = self._get_test_dir()
        test_dir.mkdir(parents=True, exist_ok=True)
        return test_dir

    def store_recording(self, request_hash: str, request: dict[str, Any], response: dict[str, Any]):
        """Store a request/response pair."""
        responses_dir = self._ensure_directory()

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
        if endpoint in ("/api/tags", "/v1/models", "/v1/openai/v1/models"):
            digest = _model_identifiers_digest(endpoint, response)
            response_file = f"models-{request_hash}-{digest}.json"

        response_path = responses_dir / response_file

        if is_debug_mode():
            logger.info("[RECORDING DEBUG] Storing recording:")
            logger.info(f"  Request hash: {request_hash}")
            logger.info(f"  File: {response_path}")
            logger.info(f"  Test ID: {get_test_context()}")
            logger.info(f"  Endpoint: {endpoint}")

        # Save response to JSON file with metadata
        with open(response_path, "w") as f:
            json.dump(
                {
                    "test_id": get_test_context(),
                    "request": request,
                    "response": serialized_response,
                    "id_normalization_mapping": {},
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

        if is_debug_mode():
            logger.info("[RECORDING DEBUG] Looking up recording:")
            logger.info(f"  Request hash: {request_hash}")
            logger.info(f"  Primary path: {response_path}")
            logger.info(f"  Primary exists: {response_path.exists()}")

        if response_path.exists():
            if is_debug_mode():
                logger.info("  Found in primary location")
            return _recording_from_file(response_path)

        # Fallback to base recordings directory (for session-level recordings)
        fallback_dir = self.base_dir / "recordings"
        fallback_path = fallback_dir / response_file

        if is_debug_mode():
            logger.info(f"  Fallback path: {fallback_path}")
            logger.info(f"  Fallback exists: {fallback_path.exists()}")

        if fallback_path.exists():
            if is_debug_mode():
                logger.info("  Found in fallback location")
            return _recording_from_file(fallback_path)

        if is_debug_mode():
            logger.info("  Recording not found in either location")

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

    mapping = data.get("id_normalization_mapping") or {}
    if mapping:
        serialized = json.dumps(data)
        for normalized, original in mapping.items():
            serialized = serialized.replace(original, normalized)
        data = json.loads(serialized)
        data["id_normalization_mapping"] = {}

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
        - '/v1/openai/v1/models' (OpenAI): response body is: [ { id: ... }, ... ]
        Returns a list of unique identifiers or None if structure doesn't match.
        """
        if "models" in response["body"]:
            # ollama
            items = response["body"]["models"]
        else:
            # openai or openai-style endpoints
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
        if endpoint in ("/v1/models", "/v1/openai/v1/models"):
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

        # Both cast(Any, ...) and type: ignore are needed here:
        # - cast(Any, ...) attempts to bypass type checking on the argument
        # - type: ignore is still needed because mypy checks the call site independently
        #   and reports arg-type mismatch even after casting
        body = ListResponse(models=cast(Any, ordered))  # type: ignore[arg-type]
    return {"request": canonical_req, "response": {"body": body, "is_streaming": False}}


async def _patched_tool_invoke_method(
    original_method, provider_name: str, self, tool_name: str, kwargs: dict[str, Any]
):
    """Patched version of tool runtime invoke_tool method for recording/replay."""
    global _current_mode, _current_storage

    if _current_mode == APIRecordingMode.LIVE or _current_storage is None:
        # Normal operation
        return await original_method(self, tool_name, kwargs)

    request_hash = normalize_tool_request(provider_name, tool_name, kwargs)

    if _current_mode in (APIRecordingMode.REPLAY, APIRecordingMode.RECORD_IF_MISSING):
        recording = _current_storage.find_recording(request_hash)
        if recording:
            return recording["response"]["body"]
        elif _current_mode == APIRecordingMode.REPLAY:
            raise RuntimeError(
                f"Recording not found for {provider_name}.{tool_name} | Request: {kwargs}\n"
                f"\n"
                f"Run './scripts/integration-tests.sh --inference-mode record-if-missing' with required API keys to generate."
            )
        # If RECORD_IF_MISSING and no recording found, fall through to record

    if _current_mode in (APIRecordingMode.RECORD, APIRecordingMode.RECORD_IF_MISSING):
        # Make the tool call and record it
        result = await original_method(self, tool_name, kwargs)

        request_data = {
            "test_id": get_test_context(),
            "provider": provider_name,
            "tool_name": tool_name,
            "kwargs": kwargs,
        }
        response_data = {"body": result, "is_streaming": False}

        # Store the recording
        _current_storage.store_recording(request_hash, request_data, response_data)
        return result

    else:
        raise AssertionError(f"Invalid mode: {_current_mode}")


async def _patched_inference_method(original_method, self, client_type, endpoint, *args, **kwargs):
    global _current_mode, _current_storage

    mode = _current_mode
    storage = _current_storage

    if is_debug_mode():
        logger.info("[RECORDING DEBUG] Entering inference method:")
        logger.info(f"  Mode: {mode}")
        logger.info(f"  Client type: {client_type}")
        logger.info(f"  Endpoint: {endpoint}")
        logger.info(f"  Test context: {get_test_context()}")

    if mode == APIRecordingMode.LIVE or storage is None:
        if endpoint in ("/v1/models", "/v1/openai/v1/models"):
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

    request_hash = normalize_inference_request(method, url, headers, body)

    # Try to find existing recording for REPLAY or RECORD_IF_MISSING modes
    recording = None
    if mode == APIRecordingMode.REPLAY or mode == APIRecordingMode.RECORD_IF_MISSING:
        # Special handling for model-list endpoints: merge all recordings with this hash
        if endpoint in ("/api/tags", "/v1/models", "/v1/openai/v1/models"):
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
        elif mode == APIRecordingMode.REPLAY:
            # REPLAY mode requires recording to exist
            if is_debug_mode():
                logger.error("[RECORDING DEBUG] Recording not found!")
                logger.error(f"  Mode: {mode}")
                logger.error(f"  Request hash: {request_hash}")
                logger.error(f"  Method: {method}")
                logger.error(f"  URL: {url}")
                logger.error(f"  Endpoint: {endpoint}")
                logger.error(f"  Model: {body.get('model', 'unknown')}")
                logger.error(f"  Test context: {get_test_context()}")
                logger.error(
                    f"  Stack config type: {os.environ.get('LLAMA_STACK_TEST_STACK_CONFIG_TYPE', 'library_client')}"
                )
            raise RuntimeError(
                f"Recording not found for request hash: {request_hash}\n"
                f"Model: {body.get('model', 'unknown')} | Request: {method} {url}\n"
                f"\n"
                f"Run './scripts/integration-tests.sh --inference-mode record-if-missing' with required API keys to generate."
            )

    if mode == APIRecordingMode.RECORD or (mode == APIRecordingMode.RECORD_IF_MISSING and not recording):
        if endpoint in ("/v1/models", "/v1/openai/v1/models"):
            response = original_method(self, *args, **kwargs)
        else:
            response = await original_method(self, *args, **kwargs)

        # we want to store the result of the iterator, not the iterator itself
        if endpoint in ("/v1/models", "/v1/openai/v1/models"):
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
            chunks: list[Any] = []
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
    """Install monkey patches for OpenAI client methods, Ollama AsyncClient methods, and tool runtime methods."""
    global _original_methods

    from ollama import AsyncClient as OllamaAsyncClient
    from openai.resources.chat.completions import AsyncCompletions as AsyncChatCompletions
    from openai.resources.completions import AsyncCompletions
    from openai.resources.embeddings import AsyncEmbeddings
    from openai.resources.models import AsyncModels

    from llama_stack.providers.remote.tool_runtime.tavily_search.tavily_search import TavilySearchToolRuntimeImpl

    # Store original methods for OpenAI, Ollama clients, and tool runtimes
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
        "tavily_invoke_tool": TavilySearchToolRuntimeImpl.invoke_tool,
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

    # Create patched methods for tool runtimes
    async def patched_tavily_invoke_tool(self, tool_name: str, kwargs: dict[str, Any]):
        return await _patched_tool_invoke_method(
            _original_methods["tavily_invoke_tool"], "tavily", self, tool_name, kwargs
        )

    # Apply tool runtime patches
    TavilySearchToolRuntimeImpl.invoke_tool = patched_tavily_invoke_tool


def unpatch_inference_clients():
    """Remove monkey patches and restore original OpenAI, Ollama client, and tool runtime methods."""
    global _original_methods

    if not _original_methods:
        return

    # Import here to avoid circular imports
    from ollama import AsyncClient as OllamaAsyncClient
    from openai.resources.chat.completions import AsyncCompletions as AsyncChatCompletions
    from openai.resources.completions import AsyncCompletions
    from openai.resources.embeddings import AsyncEmbeddings
    from openai.resources.models import AsyncModels

    from llama_stack.providers.remote.tool_runtime.tavily_search.tavily_search import TavilySearchToolRuntimeImpl

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

    # Restore tool runtime methods
    TavilySearchToolRuntimeImpl.invoke_tool = _original_methods["tavily_invoke_tool"]

    _original_methods.clear()


@contextmanager
def api_recording(mode: str, storage_dir: str | Path | None = None) -> Generator[None, None, None]:
    """Context manager for API recording/replaying (inference and tools)."""
    global _current_mode, _current_storage

    # Store previous state
    prev_mode = _current_mode
    prev_storage = _current_storage
    previous_override = None

    try:
        _current_mode = mode

        if mode in ["record", "replay", "record-if-missing"]:
            if storage_dir is None:
                raise ValueError("storage_dir is required for record, replay, and record-if-missing modes")
            _current_storage = ResponseStorage(Path(storage_dir))
            _id_counters.clear()
            patch_inference_clients()
            previous_override = set_id_override(_deterministic_id_override)

        yield

    finally:
        # Restore previous state
        if mode in ["record", "replay", "record-if-missing"]:
            unpatch_inference_clients()
            reset_id_override(previous_override)

        _current_mode = prev_mode
        _current_storage = prev_storage

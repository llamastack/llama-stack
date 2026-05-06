# Remove Library Mode from OGX

**Date:** 2026-05-06
**Status:** Draft
**Author:** Seb

## Strategic Context

OGX's moat is the server-side agentic loop. Library mode — running OGX in-process as a Python library — doesn't fit this strategy. It adds complexity, special-case code paths, and an alternative deployment model that dilutes the server-first architecture.

Library mode will be removed entirely. No deprecation period. No migration shim. Users should run the server and connect via HTTP (OpenAI SDK or ogx-client).

## What Library Mode Is

`OGXAsLibraryClient` and `AsyncOGXAsLibraryClient` in `src/ogx/core/library_client.py` (710 lines) allow users to embed OGX in-process:

```python
from ogx.core.library_client import OGXAsLibraryClient
client = OGXAsLibraryClient("starter")
response = client.models.list()
```

Instead of making HTTP requests, this loads a distribution config, initializes providers in-process, captures FastAPI route handlers, and calls them as Python functions. It requires special handling for file uploads, request injection, response conversion, event loop management, and logging.

## Scope of Removal

### Files to Delete (3)

| File | Lines | Purpose |
|------|-------|---------|
| `src/ogx/core/library_client.py` | 710 | Core library client implementation |
| `tests/unit/distribution/test_library_client_initialization.py` | 526 | Unit tests for library client |
| `docs/docs/distributions/importing_as_library.mdx` | 104 | User documentation |

### Core Code to Modify (3 files)

| File | Change |
|------|--------|
| `src/ogx/core/testing_context.py` | Change default `OGX_TEST_STACK_CONFIG_TYPE` from `"library_client"` to `"server"` |
| `src/ogx_api/common/upload_safety.py` | Remove `LibraryClientUploadFile` comments (lines 51, 57). The actual code is already generic — it uses `getattr` and `try/except TypeError` which work for any file-like object. |
| `src/ogx_api/vector_io/fastapi_routes.py` | Remove the library mode comment on line 377. The `_MISSING_REQUEST` sentinel and null-check logic is defensive coding that should stay — it handles edge cases beyond library mode. |

### Integration Test Files to Modify (19 files)

All changes follow the same pattern: remove `from ogx.core.library_client import OGXAsLibraryClient` and any `isinstance(ogx_client, OGXAsLibraryClient)` conditionals (typically skip markers).

| File | Change |
|------|--------|
| `tests/integration/conftest.py` | Remove library_client branch in config type detection |
| `tests/integration/fixtures/common.py` | Remove library client instantiation path; keep server path |
| `tests/integration/admin/test_admin.py` | Remove import and isinstance checks |
| `tests/integration/agents/test_openai_responses.py` | Remove import and isinstance checks |
| `tests/integration/responses/conftest.py` | Remove import and library mode skip logic |
| `tests/integration/responses/test_tool_responses.py` | Remove import and isinstance checks |
| `tests/integration/responses/fixtures/fixtures.py` | Remove import and isinstance checks |
| `tests/integration/inspect/test_inspect.py` | Remove import and isinstance checks |
| `tests/integration/providers/test_providers.py` | Remove import and isinstance checks |
| `tests/integration/messages/conftest.py` | Remove library_client env var default |
| `tests/integration/interactions/conftest.py` | Remove library_client env var default |
| `tests/integration/telemetry/conftest.py` | Remove library_client default |
| `tests/integration/telemetry/test_completions.py` | Remove "library_client" from span location assertion |
| `tests/integration/telemetry/collectors/base.py` | Remove "library_client" from docstring |
| `tests/integration/vector_io/test_openai_vector_stores.py` | Remove import and isinstance checks |
| `tests/integration/inference/test_openai_embeddings.py` | Remove import and isinstance checks |
| `tests/integration/inference/test_provider_data_routing.py` | Remove import and isinstance checks |
| `tests/integration/inference/test_rerank.py` | Remove import and isinstance checks |
| `tests/integration/inference/test_tools_with_schemas.py` | Remove import and isinstance checks |

### Unit Test Files to Modify (1 file)

| File | Change |
|------|--------|
| `tests/unit/files/test_upload_safety.py` | Remove `LibraryClientUploadFile` test cases |

### CI Configuration (1 file)

| File | Change |
|------|--------|
| `tests/integration/ci_matrix.json` | Remove `"allowed_clients": ["library"]` from bedrock entry. All tests run as server mode. |

### Documentation to Update (7 files)

| File | Change |
|------|--------|
| `docs/docs/distributions/index.mdx` | Remove link to importing_as_library |
| `docs/sidebars.ts` | Remove `distributions/importing_as_library` entry |
| `src/ogx/core/README.md` | Remove `library_client.py` from module listing |
| `ARCHITECTURE.md` | Remove library mode from request flow description |
| `docs/blog/2026-04-28-from-llama-stack-to-ogx.md` | Remove/update library mode mention |
| `src/ogx/providers/remote/inference/nvidia/NVIDIA.md` | Remove library_client references |
| `src/ogx/providers/remote/safety/nvidia/README.md` | Remove library_client references |

### CI Workflow (1 file)

| File | Change |
|------|--------|
| `.github/workflows/integration-tests.yml` | Remove `client: ["library", "server"]` matrix — always server. Remove conditional logic for library vs server client selection. |

## What We Keep

- **ogx-client SDK** — the HTTP client for talking to the server. This is the intended usage pattern.
- **`_MISSING_REQUEST` sentinel** in vector IO routes — defensive coding that's useful beyond library mode.
- **`OGX_TEST_STACK_CONFIG_TYPE` env var** — still useful for test infrastructure, just defaults to `"server"`.
- **Server test infrastructure** — `start_ogx_server()`, `wait_for_server_ready()`, `stop_server_on_port()`, `cleanup_server_process()` all stay as-is.

## What We Don't Do

- No deprecation warning or migration period.
- No shim that wraps a server in a library-like API.
- No changes to the ogx-client SDK.
- No changes to the server itself.

## Risk Assessment

**Low risk.** The server-mode test path is already fully functional and used in CI for every PR. The removal is primarily deleting code and simplifying conditionals.

**One flag:** The bedrock base suite (`"suite": "bedrock", "setup": "bedrock"`) is currently restricted to `"allowed_clients": ["library"]`. This means it has never run in server mode in CI. After removal, it must run in server mode. If it fails, we investigate and fix or temporarily skip it. Note that `bedrock-responses` (line 12) has no client restriction and already runs in both modes, so bedrock server-mode support exists — the base suite restriction may be historical.

## Success Criteria

1. `library_client.py` is deleted
2. No remaining imports of `OGXAsLibraryClient` or `AsyncOGXAsLibraryClient` anywhere
3. All integration tests pass in server mode
4. Pre-commit checks pass
5. Documentation no longer references library mode

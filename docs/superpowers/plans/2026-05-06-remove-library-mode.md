# Remove Library Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove OGX library mode (in-process client) to focus exclusively on the server-side deployment model.

**Architecture:** Delete `library_client.py` and all its consumers. Simplify test infrastructure to server-only. Remove library mode from docs, CI matrix, and CI workflows. The server-mode test path already exists and runs in CI — no new infrastructure needed.

**Tech Stack:** Python, pytest, GitHub Actions YAML, Docusaurus MDX

**Spec:** `docs/superpowers/specs/2026-05-06-remove-library-mode-design.md`

---

### Task 1: Delete core library client and its unit tests

**Files:**
- Delete: `src/ogx/core/library_client.py`
- Delete: `tests/unit/distribution/test_library_client_initialization.py`

- [ ] **Step 1: Delete library_client.py**

```bash
git rm src/ogx/core/library_client.py
```

- [ ] **Step 2: Delete unit tests for library client**

```bash
git rm tests/unit/distribution/test_library_client_initialization.py
```

- [ ] **Step 3: Verify no remaining imports break**

```bash
uv run python -c "from ogx.core.stack import Stack; print('core import ok')"
```

Expected: succeeds without ImportError

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -s -m "refactor: remove library_client.py and its unit tests

Remove OGXAsLibraryClient and AsyncOGXAsLibraryClient. OGX's strategic
focus is the server-side agentic loop; library mode added complexity
without fitting this direction."
```

---

### Task 2: Clean up core code referencing library mode

**Files:**
- Modify: `src/ogx/core/testing_context.py:50` — change default from `"library_client"` to `"server"`
- Modify: `src/ogx_api/common/upload_safety.py:51,57` — remove library client comments
- Modify: `src/ogx_api/vector_io/fastapi_routes.py:377` — remove library mode comment
- Modify: `src/ogx/testing/api_recorder.py:274,295,1192` — update comments and defaults
- Modify: `src/ogx/core/README.md:22` — remove library_client.py from listing

- [ ] **Step 1: Update testing_context.py default**

In `src/ogx/core/testing_context.py`, change line 50:

```python
# Before:
    stack_config_type = os.environ.get("OGX_TEST_STACK_CONFIG_TYPE", "library_client")
# After:
    stack_config_type = os.environ.get("OGX_TEST_STACK_CONFIG_TYPE", "server")
```

- [ ] **Step 2: Remove library client comments from upload_safety.py**

In `src/ogx_api/common/upload_safety.py`:

Line 51 — change:
```python
    # Use getattr because not all file-like objects have .size (e.g. LibraryClientUploadFile).
```
to:
```python
    # Use getattr because not all file-like objects have .size.
```

Line 57-58 — change:
```python
    # Some file-like objects (e.g. LibraryClientUploadFile) don't accept a size
    # argument to read(), so fall back to a single read() if chunked read fails.
```
to:
```python
    # Some file-like objects don't accept a size argument to read(), so fall
    # back to a single read() if chunked read fails.
```

- [ ] **Step 3: Remove library mode comment from vector_io routes**

In `src/ogx_api/vector_io/fastapi_routes.py`, change line 377:
```python
    # In library mode, FastAPI doesn't inject a Request.
```
to:
```python
    # Handle cases where FastAPI doesn't inject a Request.
```

- [ ] **Step 4: Update api_recorder.py comments and defaults**

In `src/ogx/testing/api_recorder.py`:

Line 274 — change:
```python
    client to server via HTTP headers. In library_client mode, this patch is a no-op
    since everything runs in the same process.
```
to:
```python
    client to server via HTTP headers.
```

Line 295 — change:
```python
        stack_config_type = os.environ.get("OGX_TEST_STACK_CONFIG_TYPE", "library_client")
```
to:
```python
        stack_config_type = os.environ.get("OGX_TEST_STACK_CONFIG_TYPE", "server")
```

Line 1192 — change:
```python
                logger.error(f"  Stack config type: {os.environ.get('OGX_TEST_STACK_CONFIG_TYPE', 'library_client')}")
```
to:
```python
                logger.error(f"  Stack config type: {os.environ.get('OGX_TEST_STACK_CONFIG_TYPE', 'server')}")
```

- [ ] **Step 5: Remove library_client.py from core README**

In `src/ogx/core/README.md`, remove line 22:
```
  library_client.py    # In-process client (no server needed)
```

- [ ] **Step 6: Run pre-commit on modified files**

```bash
uv run pre-commit run --files src/ogx/core/testing_context.py src/ogx_api/common/upload_safety.py src/ogx_api/vector_io/fastapi_routes.py src/ogx/testing/api_recorder.py src/ogx/core/README.md
```

Expected: all checks pass

- [ ] **Step 7: Commit**

```bash
git add src/ogx/core/testing_context.py src/ogx_api/common/upload_safety.py src/ogx_api/vector_io/fastapi_routes.py src/ogx/testing/api_recorder.py src/ogx/core/README.md
git commit -s -m "refactor: remove library mode references from core code

Update defaults from library_client to server, remove comments that
referenced LibraryClientUploadFile and library mode behavior."
```

---

### Task 3: Clean up integration test fixtures

**Files:**
- Modify: `tests/integration/fixtures/common.py:31,374-379,391,409`
- Modify: `tests/integration/conftest.py:56-58`

- [ ] **Step 1: Remove library client from fixtures/common.py**

In `tests/integration/fixtures/common.py`:

Remove the import on line 31:
```python
from ogx.core.library_client import OGXAsLibraryClient
```

Replace the library client fallback path (lines 336-379) in `instantiate_ogx_client()`. The current code has a fallback at the end that creates an `OGXAsLibraryClient` for non-server configs. Replace the entire block from `if "=" in config:` through `return client` (lines 336-379) with code that starts a server for these configs:

```python
    if "=" in config:
        run_config = run_config_from_dynamic_config_spec(config)

        # --stack-config bypasses template so need this to set default embedding and reranker models
        if "vector_io" in config and "inference" in config:
            embedding_model_opt = session.config.getoption("embedding_model") or ""
            # Model identifiers are in provider_id/model_id format; extract the provider.
            provider_id = embedding_model_opt.split("/")[0] if "/" in embedding_model_opt else "sentence-transformers"
            passed_model = extract_model(session.config.getoption("embedding_model"), "nomic-ai/nomic-embed-text-v1.5")
            passed_emb = session.config.getoption("embedding_dimension")

            rerank_model_opt = session.config.getoption("rerank_model") or ""
            reranker_model = None
            if rerank_model_opt:
                provider_id_of_reranker = rerank_model_opt.split("/")[0] if "/" in rerank_model_opt else "transformers"
                passed_reranker_model = extract_model(rerank_model_opt, "Qwen/Qwen3-Reranker-0.6B")
                reranker_model = RerankerModel(
                    provider_id=provider_id_of_reranker,
                    model_id=passed_reranker_model,
                )

            run_config.vector_stores = VectorStoresConfig(
                default_embedding_model=QualifiedModel(
                    provider_id=provider_id,
                    model_id=passed_model,
                    embedding_dimensions=passed_emb,
                ),
                default_reranker_model=reranker_model,
            )

        run_config_file = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
        with open(run_config_file.name, "w") as f:
            yaml.dump(run_config.model_dump(mode="json"), f)
        config = run_config_file.name
    elif "::" in config:
        # Handle distro::config.yaml format (e.g., ci-tests::run.yaml)
        config = str(resolve_config_or_distro(config))

    # Start a server for non-URL configs (distro names, file paths, dynamic specs)
    port = int(os.environ.get("OGX_PORT", DEFAULT_PORT))
    base_url = f"http://localhost:{port}"

    if is_port_available(port):
        print(f"Starting ogx server with config '{config}' on port {port}...")
        server_process = start_ogx_server(config)
        if not wait_for_server_ready(base_url, timeout=120, process=server_process):
            print("Server failed to start within timeout")
            server_process.terminate()
            raise RuntimeError(
                f"Server failed to start within timeout. Check that config '{config}' exists and is valid. "
                f"See server.log for details."
            )
        print(f"Server is ready at {base_url}")
        session._ogx_server_process = server_process
    else:
        print(f"Port {port} is already in use, assuming server is already running...")

    return OgxClient(
        base_url=base_url,
        default_headers=get_provider_data_headers(),
        timeout=int(os.environ.get("OGX_CLIENT_TIMEOUT", "30")),
    )
```

Remove the `require_server` fixture (lines 382-393) — it's no longer needed since all tests run against a server:
```python
# DELETE this entire fixture:
@pytest.fixture(scope="session")
def require_server(ogx_client):
    """..."""
    if isinstance(ogx_client, OGXAsLibraryClient):
        pytest.skip("No server running")
```

Update the `openai_client` fixture (lines 395-404) to remove the `require_server` dependency:
```python
@pytest.fixture(scope="session")
def openai_client(ogx_client):
    base_url = f"{ogx_client.base_url}/v1"
    client = OpenAI(base_url=base_url, api_key="fake", max_retries=0, timeout=30.0)
    yield client
    # Cleanup: close HTTP connections
    try:
        client.close()
    except Exception:
        pass
```

Simplify the `compat_client` fixture (lines 407-418). Remove the `isinstance` check:
```python
@pytest.fixture(params=["openai_client", "client_with_models"])
def compat_client(request, client_with_models):
    return request.getfixturevalue(request.param)
```

- [ ] **Step 2: Update conftest.py default**

In `tests/integration/conftest.py`, change the else branch (lines 56-58):

```python
# Before:
    else:
        os.environ["OGX_TEST_STACK_CONFIG_TYPE"] = "library_client"
        logger.info(f"Test stack config type: library_client (stack_config={stack_config})")
# After:
    else:
        os.environ["OGX_TEST_STACK_CONFIG_TYPE"] = "server"
        logger.info(f"Test stack config type: server (stack_config={stack_config})")
```

- [ ] **Step 3: Run pre-commit on modified files**

```bash
uv run pre-commit run --files tests/integration/fixtures/common.py tests/integration/conftest.py
```

Expected: all checks pass

- [ ] **Step 4: Commit**

```bash
git add tests/integration/fixtures/common.py tests/integration/conftest.py
git commit -s -m "refactor: remove library client from test fixtures

All integration tests now run against a real server. Remove the
OGXAsLibraryClient instantiation path, require_server fixture, and
library mode conditional in compat_client."
```

---

### Task 4: Remove library client references from integration test files (batch 1)

**Files:**
- Modify: `tests/integration/admin/test_admin.py`
- Modify: `tests/integration/inspect/test_inspect.py`
- Modify: `tests/integration/providers/test_providers.py`
- Modify: `tests/integration/responses/conftest.py`
- Modify: `tests/integration/responses/test_tool_responses.py`
- Modify: `tests/integration/responses/fixtures/fixtures.py`

For each file, the pattern is: remove `from ogx.core.library_client import OGXAsLibraryClient` and update type annotations from `OGXAsLibraryClient | OgxClient` to just `OgxClient`.

- [ ] **Step 1: Clean up admin/test_admin.py**

Remove line 9: `from ogx.core.library_client import OGXAsLibraryClient`

Change all method signatures from `ogx_client: OGXAsLibraryClient | OgxClient` to `ogx_client: OgxClient` (lines 13, 23, 28, 33, 49, 58).

- [ ] **Step 2: Clean up inspect/test_inspect.py**

Remove line 10: `from ogx.core.library_client import OGXAsLibraryClient`

Change all method signatures from `ogx_client: OGXAsLibraryClient | OgxClient` to `ogx_client: OgxClient` (lines 15, 21, 27, 46, 61).

- [ ] **Step 3: Clean up providers/test_providers.py**

Remove line 9: `from ogx.core.library_client import OGXAsLibraryClient`

Change line 13 from `ogx_client: OGXAsLibraryClient | OgxClient` to `ogx_client: OgxClient`.

- [ ] **Step 4: Clean up responses/conftest.py**

Remove line 11: `from ogx.core.library_client import OGXAsLibraryClient`

Remove the library mode skip block around line 36:
```python
# Remove this block:
    if isinstance(compat_client, OGXAsLibraryClient):
        pytest.skip("...")
```

- [ ] **Step 5: Clean up responses/test_tool_responses.py**

Remove line 17: `from ogx.core.library_client import OGXAsLibraryClient`

Remove the isinstance check around line 269:
```python
# Remove this conditional:
        if isinstance(responses_client, OGXAsLibraryClient)
```

- [ ] **Step 6: Clean up responses/fixtures/fixtures.py**

Remove line 14: `from ogx.core.library_client import OGXAsLibraryClient`

In the `openai_client` fixture (lines 110-116), remove the `stack:` config branch that creates an `OGXAsLibraryClient`:
```python
# Remove this block:
    if provider.startswith("stack:"):
        parts = provider.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid config for OGX: {provider}, it must be of the form 'stack:<config>'")
        config = parts[1]
        client = OGXAsLibraryClient(config, skip_logger_removal=True)
        return client
```

- [ ] **Step 7: Run pre-commit on all modified files**

```bash
uv run pre-commit run --files tests/integration/admin/test_admin.py tests/integration/inspect/test_inspect.py tests/integration/providers/test_providers.py tests/integration/responses/conftest.py tests/integration/responses/test_tool_responses.py tests/integration/responses/fixtures/fixtures.py
```

- [ ] **Step 8: Commit**

```bash
git add tests/integration/admin/ tests/integration/inspect/ tests/integration/providers/ tests/integration/responses/
git commit -s -m "refactor: remove library client references from test files (batch 1)

Remove OGXAsLibraryClient imports, type annotations, and isinstance
checks from admin, inspect, providers, and responses test files."
```

---

### Task 5: Remove library client references from integration test files (batch 2)

**Files:**
- Modify: `tests/integration/agents/test_openai_responses.py`
- Modify: `tests/integration/messages/conftest.py`
- Modify: `tests/integration/interactions/conftest.py`
- Modify: `tests/integration/vector_io/test_openai_vector_stores.py`
- Modify: `tests/integration/inference/test_openai_embeddings.py`
- Modify: `tests/integration/inference/test_provider_data_routing.py`
- Modify: `tests/integration/inference/test_rerank.py`
- Modify: `tests/integration/inference/test_tools_with_schemas.py`

- [ ] **Step 1: Clean up agents/test_openai_responses.py**

Remove line 9: `from ogx.core.library_client import OGXAsLibraryClient`

Remove the isinstance skip blocks at lines 126, 219, 271, 473. Each looks like:
```python
        if isinstance(client_with_models, OGXAsLibraryClient):
            pytest.skip("...")
```
Remove these conditionals entirely — the tests now always run against a server.

- [ ] **Step 2: Clean up messages/conftest.py**

Remove line 14: `from ogx.core.library_client import OGXAsLibraryClient`

Remove the isinstance check at line 32:
```python
    if isinstance(ogx_client, OGXAsLibraryClient):
        ...
```

- [ ] **Step 3: Clean up interactions/conftest.py**

Remove line 14: `from ogx.core.library_client import OGXAsLibraryClient`

Remove the isinstance check at line 32:
```python
    if isinstance(ogx_client, OGXAsLibraryClient):
        ...
```

Update line 41 default from `"library_client"` to `"server"`:
```python
    stack_config_type = os.environ.get("OGX_TEST_STACK_CONFIG_TYPE", "server")
```

- [ ] **Step 4: Clean up vector_io/test_openai_vector_stores.py**

Remove line 15: `from ogx.core.library_client import OGXAsLibraryClient`

Remove isinstance checks at lines 4504, 5185, 5197. Each is a skip block like:
```python
        if isinstance(compat_client, OGXAsLibraryClient):
            pytest.skip("...")
```

- [ ] **Step 5: Clean up inference/test_openai_embeddings.py**

Remove line 13: `from ogx.core.library_client import OGXAsLibraryClient`

Remove the isinstance check at line 131:
```python
        if request.param == "openai_client" and isinstance(client_with_models, OGXAsLibraryClient):
```
This was a compat_client fixture equivalent — now that we always have a server, simplify to just check param name.

- [ ] **Step 6: Clean up inference/test_provider_data_routing.py**

Remove line 19: `from ogx.core.library_client import OGXAsLibraryClient`

Remove the isinstance check at line 43:
```python
        if not isinstance(client_with_models, OGXAsLibraryClient):
```

- [ ] **Step 7: Clean up inference/test_rerank.py**

Remove line 17: `from ogx.core.library_client import OGXAsLibraryClient`

At line 128, remove the conditional error type selection:
```python
# Before:
        error_type = ValueError if isinstance(client_with_models, OGXAsLibraryClient) else OGXBadRequestError
# After:
        error_type = OGXBadRequestError
```

- [ ] **Step 8: Clean up inference/test_tools_with_schemas.py**

Remove line 14: `from ogx.core.library_client import OGXAsLibraryClient`

Remove the isinstance check at line 161:
```python
        if not isinstance(ogx_client, OGXAsLibraryClient):
```

- [ ] **Step 9: Run pre-commit on all modified files**

```bash
uv run pre-commit run --files tests/integration/agents/test_openai_responses.py tests/integration/messages/conftest.py tests/integration/interactions/conftest.py tests/integration/vector_io/test_openai_vector_stores.py tests/integration/inference/test_openai_embeddings.py tests/integration/inference/test_provider_data_routing.py tests/integration/inference/test_rerank.py tests/integration/inference/test_tools_with_schemas.py
```

- [ ] **Step 10: Commit**

```bash
git add tests/integration/agents/ tests/integration/messages/ tests/integration/interactions/ tests/integration/vector_io/ tests/integration/inference/
git commit -s -m "refactor: remove library client references from test files (batch 2)

Remove OGXAsLibraryClient imports and isinstance conditionals from
agents, messages, interactions, vector_io, and inference test files."
```

---

### Task 6: Clean up telemetry tests

**Files:**
- Modify: `tests/integration/telemetry/conftest.py:21`
- Modify: `tests/integration/telemetry/test_completions.py:98`
- Modify: `tests/integration/telemetry/collectors/base.py:87`

- [ ] **Step 1: Update telemetry conftest.py default**

Change line 21:
```python
# Before:
    stack_mode = os.environ.get("OGX_TEST_STACK_CONFIG_TYPE", "library_client")
# After:
    stack_mode = os.environ.get("OGX_TEST_STACK_CONFIG_TYPE", "server")
```

- [ ] **Step 2: Update test_completions.py assertion**

Change line 98:
```python
# Before:
        assert span.get_location() in ["library_client", "server"]
# After:
        assert span.get_location() == "server"
```

- [ ] **Step 3: Update collectors/base.py docstring**

Change line 87:
```python
# Before:
        """Get the location (library_client, server) for root spans."""
# After:
        """Get the location for root spans."""
```

- [ ] **Step 4: Run pre-commit**

```bash
uv run pre-commit run --files tests/integration/telemetry/conftest.py tests/integration/telemetry/test_completions.py tests/integration/telemetry/collectors/base.py
```

- [ ] **Step 5: Commit**

```bash
git add tests/integration/telemetry/
git commit -s -m "refactor: remove library_client from telemetry test infrastructure

Update defaults and assertions to server-only mode."
```

---

### Task 7: Update CI configuration

**Files:**
- Modify: `tests/integration/ci_matrix.json:4`
- Modify: `.github/workflows/integration-tests.yml:134,196-200`
- Modify: `scripts/integration-tests.sh:206-207,211,226`

- [ ] **Step 1: Remove library-only restriction from ci_matrix.json**

Remove `"allowed_clients": ["library"]` from the bedrock entry (line 4):

```json
// Before:
    {"suite": "bedrock", "setup": "bedrock", "allowed_clients": ["library"]},
// After:
    {"suite": "bedrock", "setup": "bedrock"},
```

- [ ] **Step 2: Simplify CI workflow matrix**

In `.github/workflows/integration-tests.yml`, change line 134:

```yaml
# Before:
        client: ${{ github.event_name == 'pull_request' && fromJSON('["server"]') || fromJSON('["library", "server"]') }}
# After:
        client: ["server"]
```

Simplify the stack-config selection (lines 196-200):

```yaml
# Before:
          stack-config: >-
            ${{ matrix.config.stack_config
                || (matrix.client == 'library' && 'ci-tests')
                || (matrix.client == 'server' && 'server:ci-tests')
                || 'docker:ci-tests' }}
# After:
          stack-config: >-
            ${{ matrix.config.stack_config
                || 'server:ci-tests' }}
```

- [ ] **Step 3: Update integration-tests.sh**

In `scripts/integration-tests.sh`, change the else branch (lines 205-207):

```bash
# Before:
    else
        export OGX_TEST_STACK_CONFIG_TYPE="library_client"
        echo "Setting stack config type: library_client"
# After:
    else
        export OGX_TEST_STACK_CONFIG_TYPE="server"
        echo "Setting stack config type: server"
```

Update the comment on line 211:
```bash
# Before:
    # - For library client and server mode: localhost (both on same host)
# After:
    # - For server mode: localhost (on same host)
```

Update the echo on line 226:
```bash
# Before:
        echo "Setting MCP host: localhost (library/server mode)"
# After:
        echo "Setting MCP host: localhost (server mode)"
```

- [ ] **Step 4: Commit**

```bash
git add tests/integration/ci_matrix.json .github/workflows/integration-tests.yml scripts/integration-tests.sh
git commit -s -m "ci: remove library client mode from CI matrix and workflows

All integration tests now run in server mode only. Remove the library
client matrix dimension and simplify stack-config selection."
```

---

### Task 8: Update documentation

**Files:**
- Delete: `docs/docs/distributions/importing_as_library.mdx`
- Modify: `docs/docs/distributions/index.mdx:61-68`
- Modify: `docs/sidebars.ts:100`
- Modify: `docs/docs/distributions/starting_ogx_server.mdx:51-64`
- Modify: `docs/docs/getting_started/detailed_tutorial.mdx:205-221`
- Modify: `docs/blog/2026-04-28-from-llama-stack-to-ogx.md:113`
- Modify: `src/ogx/providers/remote/inference/nvidia/NVIDIA.md:38-42`
- Modify: `src/ogx/providers/remote/safety/nvidia/README.md:36-40`

- [ ] **Step 1: Delete importing_as_library.mdx**

```bash
git rm docs/docs/distributions/importing_as_library.mdx
```

- [ ] **Step 2: Remove library link from distributions/index.mdx**

Remove lines 61-68 (the "Importing as Library" card):
```mdx
  {
    type: 'link',
    label: 'Importing as Library',
    href: '/docs/distributions/importing_as_library',
    description: 'Use distributions programmatically in your Python code',
  },
```

- [ ] **Step 3: Remove library entry from sidebars.ts**

Remove line 100:
```typescript
        'distributions/importing_as_library',
```

- [ ] **Step 4: Remove library tab from starting_ogx_server.mdx**

Remove the entire library TabItem (lines 51-64):
```mdx
<TabItem value="library" label="As a Library">

Use OGX directly in your Python process without running a server:

```python
from ogx.core.library_client import OGXAsLibraryClient

client = OGXAsLibraryClient("starter")
client.initialize()
```

See [Using OGX as a Library](importing_as_library) for details.

</TabItem>
```

- [ ] **Step 5: Remove library section from detailed_tutorial.mdx**

Remove lines 205-221 (the "Running as a Library" section):
```mdx
### Running as a Library

You can also use OGX without running a server, directly in your Python process:

```python title="library.py"
from ogx.core.library_client import OGXAsLibraryClient

client = OGXAsLibraryClient("starter")
client.initialize()

# Use the same OpenAI-compatible interface
response = client.responses.create(
    model="ollama/llama3.2:3b",
    input="Hello from library mode!",
)
print(response.output_text)
```
```

- [ ] **Step 6: Update blog post**

In `docs/blog/2026-04-28-from-llama-stack-to-ogx.md`, change line 113:

```markdown
# Before:
**Server-first, not a framework.** The primary deployment model is an HTTP server that your application talks to over the network, in any language. A library mode exists for embedding OGX in-process, but the architecture is designed around the server: pluggable providers, API translation, and agentic orchestration all happen on the server side.
# After:
**Server-first, not a framework.** The primary deployment model is an HTTP server that your application talks to over the network, in any language. Pluggable providers, API translation, and agentic orchestration all happen on the server side.
```

- [ ] **Step 7: Update NVIDIA inference provider docs**

In `src/ogx/providers/remote/inference/nvidia/NVIDIA.md`, replace the library client initialization block (lines 38-42) with server-based usage:

```python
# After starting the server with `ogx stack run nvidia`:
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="empty")
```

- [ ] **Step 8: Update NVIDIA safety provider docs**

In `src/ogx/providers/remote/safety/nvidia/README.md`, replace the library client initialization block (lines 36-40) with server-based usage:

```python
# After starting the server with `ogx stack run nvidia`:
from ogx_client import OgxClient

client = OgxClient(base_url="http://localhost:8321")
```

- [ ] **Step 9: Run pre-commit on modified doc files**

```bash
uv run pre-commit run --files docs/docs/distributions/index.mdx docs/sidebars.ts docs/docs/distributions/starting_ogx_server.mdx docs/docs/getting_started/detailed_tutorial.mdx docs/blog/2026-04-28-from-llama-stack-to-ogx.md src/ogx/providers/remote/inference/nvidia/NVIDIA.md src/ogx/providers/remote/safety/nvidia/README.md
```

- [ ] **Step 10: Commit**

```bash
git add docs/ src/ogx/providers/remote/inference/nvidia/NVIDIA.md src/ogx/providers/remote/safety/nvidia/README.md
git commit -s -m "docs: remove all library mode documentation

Delete importing_as_library page, remove library tabs and sections from
server and tutorial docs, update provider docs to use server-based
initialization, and update blog post."
```

---

### Task 9: Final verification

- [ ] **Step 1: Verify no remaining library_client references**

```bash
grep -r "library_client\|OGXAsLibraryClient\|AsyncOGXAsLibraryClient\|LibraryClientUploadFile\|LibraryClientHttpxResponse\|importing_as_library" --include="*.py" --include="*.yml" --include="*.yaml" --include="*.mdx" --include="*.md" --include="*.ts" --include="*.json" src/ tests/ docs/ .github/ scripts/
```

Expected: no output (zero matches)

- [ ] **Step 2: Run pre-commit on all files**

```bash
uv run pre-commit run --all-files
```

Expected: all checks pass

- [ ] **Step 3: Run unit tests**

```bash
uv run pytest tests/unit/ -x --tb=short -q
```

Expected: all pass, no import errors

- [ ] **Step 4: Verify the grep found nothing**

If Step 1 found any remaining references, go back and clean them up. Then re-run Steps 1-3.

# server

FastAPI server implementation for Llama Stack.

## Directory Structure

```
server/
  __init__.py
  server.py                    # Main FastAPI app, route dispatch, SSE streaming, lifespan
  auth.py                      # AuthenticationMiddleware (Bearer token validation)
  auth_providers.py            # Auth provider implementations (Kubernetes, custom endpoint)
  quota.py                     # QuotaMiddleware (rate limiting per client)
  routes.py                    # Route discovery from @webmethod protocols
  fastapi_router_registry.py   # FastAPI router registry for migrated APIs
```

## How It Works

### Server Startup

1. `main()` in `server.py` resolves the config, creates a `StackApp` (subclass of `FastAPI`).
2. `StackApp.__init__` creates and initializes a `Stack` instance (provider resolution, resource registration).
3. The lifespan context starts background tasks (e.g., periodic registry refresh).

### Route Registration

Routes come from two sources:

- **Legacy `@webmethod` routes**: Discovered by `get_all_api_routes()` in `routes.py`, which inspects protocol methods for `@webmethod` decorators.
- **FastAPI router routes**: Registered via `fastapi_router_registry.py` for APIs that have been migrated to native FastAPI routers.

### Middleware

- **`AuthenticationMiddleware`** (`auth.py`): Validates Bearer tokens using a configured auth provider (Kubernetes, custom endpoint). Extracts user identity and attributes for access control. Endpoints can opt out with `require_authentication=False`.
- **`QuotaMiddleware`** (`quota.py`): Enforces per-client rate limits (separate limits for authenticated vs. anonymous). Uses KVStore for tracking request counts.

### Response Handling

- Non-streaming responses return JSON via FastAPI's standard response handling.
- Streaming responses use Server-Sent Events (SSE) via `StreamingResponse`, with `create_sse_event()` serializing each chunk.
- Exceptions are translated to appropriate HTTP status codes by `translate_exception()`.

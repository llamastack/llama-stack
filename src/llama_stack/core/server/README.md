# server

FastAPI server implementation for Llama Stack.

## Directory Structure

```text
server/
  __init__.py
  server.py                    # Main FastAPI app, middleware, lifespan
  auth.py                      # AuthenticationMiddleware (Bearer token validation)
  auth_providers.py            # Auth provider implementations (Kubernetes, custom endpoint)
  quota.py                     # QuotaMiddleware (rate limiting per client)
  metrics.py                   # RequestMetricsMiddleware (OpenTelemetry counters)
  routes.py                    # Route-to-auth-info mapping for middleware
  fastapi_router_registry.py   # Auto-discovery of FastAPI routers from llama_stack_api
```

## How It Works

### Server Startup

1. `main()` in `server.py` resolves the config, creates a `StackApp` (subclass of `FastAPI`).
2. `StackApp.__init__` creates and initializes a `Stack` instance (provider resolution, resource registration).
3. The lifespan context starts background tasks (e.g., periodic registry refresh).

### Route Registration

Routes are auto-discovered from `llama_stack_api` packages by `fastapi_router_registry.py`. Each API package that has a `fastapi_routes` submodule with a `create_router()` factory is registered at startup. External APIs can also provide routers via `register_external_api_routers()`.

### Middleware

- **`AuthenticationMiddleware`** (`auth.py`): Validates Bearer tokens using a configured auth provider (Kubernetes, custom endpoint). Extracts user identity and attributes for access control. Routes can opt out via the `PUBLIC_ROUTE_KEY` in their `openapi_extra`.
- **`QuotaMiddleware`** (`quota.py`): Enforces per-client rate limits (separate limits for authenticated vs. anonymous). Uses KVStore for tracking request counts.
- **`RequestMetricsMiddleware`** (`metrics.py`): Tracks request counts, durations, and concurrent requests via OpenTelemetry.

### Response Handling

- Non-streaming responses return JSON via FastAPI's standard response handling.
- Streaming responses use Server-Sent Events (SSE) via `StreamingResponse`, with SSE formatting handled per-router in the `llama_stack_api` package.
- Exceptions are translated to appropriate HTTP status codes by `translate_exception()`.

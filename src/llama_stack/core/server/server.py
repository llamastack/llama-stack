# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import concurrent.futures
import inspect
import json
import os
import sys
import traceback
import warnings
from contextlib import asynccontextmanager
from importlib.metadata import version as parse_version
from pathlib import Path
from typing import Any

import httpx
import rich.pretty
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import BadRequestError, NotFoundError
from pydantic import BaseModel, ValidationError

from llama_stack.apis.common.errors import ConflictError, ResourceNotFoundError
from llama_stack.core.access_control.access_control import AccessDeniedError
from llama_stack.core.datatypes import (
    AuthenticationRequiredError,
    StackRunConfig,
    process_cors_config,
)
from llama_stack.core.distribution import builtin_automatically_routed_apis
from llama_stack.core.external import load_external_apis
from llama_stack.core.stack import (
    Stack,
    cast_image_name_to_string,
    replace_env_vars,
)
from llama_stack.core.telemetry import Telemetry
from llama_stack.core.telemetry.tracing import setup_logger
from llama_stack.core.utils.config import redact_sensitive_fields
from llama_stack.core.utils.config_resolution import Mode, resolve_config_or_distro
from llama_stack.log import LoggingConfig, get_logger, setup_logging
from llama_stack.providers.datatypes import Api

from .auth import AuthenticationMiddleware
from .quota import QuotaMiddleware
from .tracing import TracingMiddleware

REPO_ROOT = Path(__file__).parent.parent.parent.parent

logger = get_logger(name=__name__, category="core::server")


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, "write") else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


if os.environ.get("LLAMA_STACK_TRACE_WARNINGS"):
    warnings.showwarning = warn_with_traceback


def create_sse_event(data: Any) -> str:
    if isinstance(data, BaseModel):
        data = data.model_dump_json()
    else:
        data = json.dumps(data)

    return f"data: {data}\n\n"


async def global_exception_handler(request: Request, exc: Exception):
    http_exc = translate_exception(exc)

    # Only log full tracebacks for unexpected server errors (5xx), not expected client/provider errors
    if http_exc.status_code >= 500:
        traceback.print_exception(exc)
    elif http_exc.status_code >= 400:
        # Log expected errors at debug level without full traceback
        logger.debug(f"Client error {http_exc.status_code}: {http_exc.detail}")

    return JSONResponse(status_code=http_exc.status_code, content={"error": {"detail": http_exc.detail}})


def translate_exception(exc: Exception) -> HTTPException | RequestValidationError:
    if isinstance(exc, ValidationError):
        exc = RequestValidationError(exc.errors())

    if isinstance(exc, RequestValidationError):
        return HTTPException(
            status_code=httpx.codes.BAD_REQUEST,
            detail={
                "errors": [
                    {
                        "loc": list(error["loc"]),
                        "msg": error["msg"],
                        "type": error["type"],
                    }
                    for error in exc.errors()
                ]
            },
        )
    elif isinstance(exc, ConflictError):
        return HTTPException(status_code=httpx.codes.CONFLICT, detail=str(exc))
    elif isinstance(exc, ResourceNotFoundError):
        return HTTPException(status_code=httpx.codes.NOT_FOUND, detail=str(exc))
    elif isinstance(exc, ValueError):
        return HTTPException(status_code=httpx.codes.BAD_REQUEST, detail=f"Invalid value: {str(exc)}")
    elif isinstance(exc, BadRequestError):
        return HTTPException(status_code=httpx.codes.BAD_REQUEST, detail=str(exc))
    elif isinstance(exc, NotFoundError):
        return HTTPException(status_code=httpx.codes.NOT_FOUND, detail=str(exc))
    elif isinstance(exc, PermissionError | AccessDeniedError):
        return HTTPException(status_code=httpx.codes.FORBIDDEN, detail=f"Permission denied: {str(exc)}")
    elif isinstance(exc, ConnectionError | httpx.ConnectError):
        return HTTPException(status_code=httpx.codes.BAD_GATEWAY, detail=str(exc))
    elif isinstance(exc, asyncio.TimeoutError | TimeoutError):
        return HTTPException(status_code=httpx.codes.GATEWAY_TIMEOUT, detail=f"Operation timed out: {str(exc)}")
    elif isinstance(exc, NotImplementedError):
        return HTTPException(status_code=httpx.codes.NOT_IMPLEMENTED, detail=f"Not implemented: {str(exc)}")
    elif isinstance(exc, AuthenticationRequiredError):
        return HTTPException(status_code=httpx.codes.UNAUTHORIZED, detail=f"Authentication required: {str(exc)}")
    elif hasattr(exc, "status_code") and isinstance(getattr(exc, "status_code", None), int):
        # Handle provider SDK exceptions (e.g., OpenAI's APIStatusError and subclasses)
        # These include AuthenticationError (401), PermissionDeniedError (403), etc.
        # This preserves the actual HTTP status code from the provider
        status_code = exc.status_code
        detail = str(exc)
        return HTTPException(status_code=status_code, detail=detail)
    else:
        return HTTPException(
            status_code=httpx.codes.INTERNAL_SERVER_ERROR,
            detail="Internal server error: An unexpected error occurred.",
        )


class StackApp(FastAPI):
    """
    A wrapper around the FastAPI application to hold a reference to the Stack instance so that we can
    start background tasks (e.g. refresh model registry periodically) from the lifespan context manager.
    """

    def __init__(self, config: StackRunConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stack: Stack = Stack(config)

        # This code is called from a running event loop managed by uvicorn so we cannot simply call
        # asyncio.run() to initialize the stack. We cannot await either since this is not an async
        # function.
        # As a workaround, we use a thread pool executor to run the initialize() method
        # in a separate thread.
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, self.stack.initialize())
            future.result()


@asynccontextmanager
async def lifespan(app: StackApp):
    server_version = parse_version("llama-stack")

    logger.info(f"Starting up Llama Stack server (version: {server_version})")
    assert app.stack is not None
    app.stack.create_registry_refresh_task()
    yield
    logger.info("Shutting down")
    await app.stack.shutdown()


async def maybe_await(value):
    if inspect.iscoroutine(value):
        return await value
    return value


async def sse_generator(event_gen_coroutine):
    event_gen = None
    try:
        event_gen = await event_gen_coroutine
        async for item in event_gen:
            yield create_sse_event(item)
    except asyncio.CancelledError:
        logger.info("Generator cancelled")
        if event_gen:
            await event_gen.aclose()
    except Exception as e:
        logger.exception("Error in sse_generator")
        yield create_sse_event(
            {
                "error": {
                    "message": str(translate_exception(e)),
                },
            }
        )


async def log_request_pre_validation(request: Request):
    if request.method in ("POST", "PUT", "PATCH"):
        try:
            body_bytes = await request.body()
            if body_bytes:
                try:
                    parsed_body = json.loads(body_bytes.decode())
                    log_output = rich.pretty.pretty_repr(parsed_body)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    log_output = repr(body_bytes)
                logger.debug(f"Incoming raw request body for {request.method} {request.url.path}:\n{log_output}")
            else:
                logger.debug(f"Incoming {request.method} {request.url.path} request with empty body.")
        except Exception as e:
            logger.warning(f"Could not read or log request body for {request.method} {request.url.path}: {e}")


class ClientVersionMiddleware:
    def __init__(self, app):
        self.app = app
        self.server_version = parse_version("llama-stack")

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            client_version = headers.get(b"x-llamastack-client-version", b"").decode()
            if client_version:
                try:
                    client_version_parts = tuple(map(int, client_version.split(".")[:2]))
                    server_version_parts = tuple(map(int, self.server_version.split(".")[:2]))
                    if client_version_parts != server_version_parts:

                        async def send_version_error(send):
                            await send(
                                {
                                    "type": "http.response.start",
                                    "status": httpx.codes.UPGRADE_REQUIRED,
                                    "headers": [[b"content-type", b"application/json"]],
                                }
                            )
                            error_msg = json.dumps(
                                {
                                    "error": {
                                        "message": f"Client version {client_version} is not compatible with server version {self.server_version}. Please update your client."
                                    }
                                }
                            ).encode()
                            await send({"type": "http.response.body", "body": error_msg})

                        return await send_version_error(send)
                except (ValueError, IndexError):
                    # If version parsing fails, let the request through
                    pass

        return await self.app(scope, receive, send)


def create_app() -> StackApp:
    """Create and configure the FastAPI application.

    This factory function reads configuration from environment variables:
    - LLAMA_STACK_CONFIG: Path to config file (required)

    Returns:
        Configured StackApp instance.
    """
    # Initialize logging from environment variables first
    setup_logging()

    config_file = os.getenv("LLAMA_STACK_CONFIG")
    if config_file is None:
        raise ValueError("LLAMA_STACK_CONFIG environment variable is required")

    config_file = resolve_config_or_distro(config_file, Mode.RUN)

    # Load and process configuration
    logger_config = None
    with open(config_file) as fp:
        config_contents = yaml.safe_load(fp)
        if isinstance(config_contents, dict) and (cfg := config_contents.get("logging_config")):
            logger_config = LoggingConfig(**cfg)
        logger = get_logger(name=__name__, category="core::server", config=logger_config)

        config = replace_env_vars(config_contents)
        config = StackRunConfig(**cast_image_name_to_string(config))

    _log_run_config(run_config=config)

    app = StackApp(
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        config=config,
    )

    if not os.environ.get("LLAMA_STACK_DISABLE_VERSION_CHECK"):
        app.add_middleware(ClientVersionMiddleware)

    impls = app.stack.impls

    if config.server.auth:
        logger.info(f"Enabling authentication with provider: {config.server.auth.provider_config.type.value}")
        app.add_middleware(AuthenticationMiddleware, auth_config=config.server.auth, impls=impls)
    else:
        if config.server.quota:
            quota = config.server.quota
            logger.warning(
                "Configured authenticated_max_requests (%d) but no auth is enabled; "
                "falling back to anonymous_max_requests (%d) for all the requests",
                quota.authenticated_max_requests,
                quota.anonymous_max_requests,
            )

    if config.server.quota:
        logger.info("Enabling quota middleware for authenticated and anonymous clients")

        quota = config.server.quota
        anonymous_max_requests = quota.anonymous_max_requests
        # if auth is disabled, use the anonymous max requests
        authenticated_max_requests = quota.authenticated_max_requests if config.server.auth else anonymous_max_requests

        kv_config = quota.kvstore
        window_map = {"day": 86400}
        window_seconds = window_map[quota.period.value]

        app.add_middleware(
            QuotaMiddleware,
            kv_config=kv_config,
            anonymous_max_requests=anonymous_max_requests,
            authenticated_max_requests=authenticated_max_requests,
            window_seconds=window_seconds,
        )

    if config.server.cors:
        logger.info("Enabling CORS")
        cors_config = process_cors_config(config.server.cors)
        if cors_config:
            app.add_middleware(CORSMiddleware, **cors_config.model_dump())

    if config.telemetry.enabled:
        setup_logger(Telemetry())

    # Load external APIs if configured
    external_apis = load_external_apis(config)

    # Import API modules to ensure routers are registered (they register on import)
    # External APIs must also register their routers via register_router() in their module
    from llama_stack.apis import (  # noqa: F401
        batches,
        benchmarks,
        conversations,
        datasetio,
        datasets,
        eval,
        files,
        inspect,
        models,
        prompts,
        providers,
        safety,
        scoring,
        scoring_functions,
        shields,
        tools,
    )

    # Import router registry to ensure routers are registered
    from llama_stack.core.server.routers import create_router, has_router

    if config.apis:
        apis_to_serve = set(config.apis)
    else:
        apis_to_serve = set(impls.keys())

    for inf in builtin_automatically_routed_apis():
        # if we do not serve the corresponding router API, we should not serve the routing table API
        if inf.router_api.value not in apis_to_serve:
            continue
        apis_to_serve.add(inf.routing_table_api.value)

    apis_to_serve.add("inspect")
    apis_to_serve.add("providers")
    apis_to_serve.add("prompts")
    apis_to_serve.add("conversations")

    # Store impls in app state for dependency injection
    app.state.impls = impls

    # Create impl_getter function for routers
    def impl_getter(api: Api) -> Any:
        return impls[api]

    # Register routers for all APIs (including external APIs that must register routers)
    for api_str in apis_to_serve:
        api = Api(api_str)
        if has_router(api):
            router = create_router(api, impl_getter)
            if router:
                app.include_router(router)
                logger.debug(f"Registered router for {api} API")
        else:
            # All APIs must now use routers - this is a breaking change for external APIs
            logger.warning(
                f"API '{api.value}' does not have a registered router. "
                f"External APIs must register a router using register_router() in their module. "
                f"Skipping this API."
            )

    logger.debug(f"serving APIs: {apis_to_serve}")

    app.exception_handler(RequestValidationError)(global_exception_handler)
    app.exception_handler(Exception)(global_exception_handler)

    if config.telemetry.enabled:
        app.add_middleware(TracingMiddleware, impls=impls, external_apis=external_apis)

    return app


def _log_run_config(run_config: StackRunConfig):
    """Logs the run config with redacted fields and disabled providers removed."""
    logger.info("Run configuration:")
    safe_config = redact_sensitive_fields(run_config.model_dump(mode="json"))
    clean_config = remove_disabled_providers(safe_config)
    logger.info(yaml.dump(clean_config, indent=2))


def remove_disabled_providers(obj):
    if isinstance(obj, dict):
        keys = ["provider_id", "shield_id", "provider_model_id", "model_id"]
        if any(k in obj and obj[k] in ("__disabled__", "", None) for k in keys):
            return None
        return {k: v for k, v in ((k, remove_disabled_providers(v)) for k, v in obj.items()) if v is not None}
    elif isinstance(obj, list):
        return [item for item in (remove_disabled_providers(i) for i in obj) if item is not None]
    else:
        return obj

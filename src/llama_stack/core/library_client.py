# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import inspect
import json
import logging  # allow-direct-logging
import os
import sys
import typing
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin

import httpx
import yaml
from fastapi import Response as FastAPIResponse

from llama_stack.core.utils.type_inspection import is_unwrapped_body_param

try:
    from llama_stack_client import (
        NOT_GIVEN,
        APIResponse,
        AsyncAPIResponse,
        AsyncLlamaStackClient,
        AsyncStream,
        LlamaStackClient,
    )
except ImportError as e:
    raise ImportError(
        "llama-stack-client is not installed. Please install it with `uv pip install llama-stack[client]`."
    ) from e

from pydantic import BaseModel, TypeAdapter
from rich.console import Console
from termcolor import cprint

from llama_stack.core.build import print_pip_install_help
from llama_stack.core.configure import parse_and_maybe_upgrade_config
from llama_stack.core.request_headers import PROVIDER_DATA_VAR, request_provider_data_context
from llama_stack.core.resolver import ProviderRegistry
from llama_stack.core.server.routes import RouteImpls, find_matching_route, initialize_route_impls
from llama_stack.core.stack import Stack, get_stack_run_config_from_distro, replace_env_vars
from llama_stack.core.utils.config import redact_sensitive_fields
from llama_stack.core.utils.context import preserve_contexts_async_generator
from llama_stack.core.utils.exec import in_notebook
from llama_stack.log import get_logger, setup_logging

logger = get_logger(name=__name__, category="core")

T = TypeVar("T")


def convert_pydantic_to_json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    elif isinstance(value, list):
        return [convert_pydantic_to_json_value(item) for item in value]
    elif isinstance(value, dict):
        return {k: convert_pydantic_to_json_value(v) for k, v in value.items()}
    elif isinstance(value, BaseModel):
        return json.loads(value.model_dump_json())
    else:
        return value


def convert_to_pydantic(annotation: Any, value: Any) -> Any:
    if isinstance(annotation, type) and annotation in {str, int, float, bool}:
        return value

    origin = get_origin(annotation)

    if origin is list:
        item_type = get_args(annotation)[0]
        try:
            return [convert_to_pydantic(item_type, item) for item in value]
        except Exception:
            logger.error(f"Error converting list {value} into {item_type}")
            return value

    elif origin is dict:
        key_type, val_type = get_args(annotation)
        try:
            return {k: convert_to_pydantic(val_type, v) for k, v in value.items()}
        except Exception:
            logger.error(f"Error converting dict {value} into {val_type}")
            return value

    try:
        # Handle Pydantic models and discriminated unions
        return TypeAdapter(annotation).validate_python(value)

    except Exception as e:
        # TODO: this is workardound for having Union[str, AgentToolGroup] in API schema.
        # We should get rid of any non-discriminated unions in the API schema.
        if origin is Union:
            for union_type in get_args(annotation):
                try:
                    return convert_to_pydantic(union_type, value)
                except Exception:
                    continue
            logger.warning(
                f"Warning: direct client failed to convert parameter {value} into {annotation}: {e}",
            )
        raise ValueError(f"Failed to convert parameter {value} into {annotation}: {e}") from e


class LibraryClientUploadFile:
    """LibraryClient UploadFile object that mimics FastAPI's UploadFile interface."""

    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self.content = content
        self.content_type = "application/octet-stream"

    async def read(self) -> bytes:
        return self.content


class LibraryClientHttpxResponse:
    """LibraryClient httpx Response object for FastAPI Response conversion."""

    def __init__(self, response):
        self.content = response.body if isinstance(response.body, bytes) else response.body.encode()
        self.status_code = response.status_code
        self.headers = response.headers


class LlamaStackAsLibraryClient(LlamaStackClient):
    def __init__(
        self,
        config_path_or_distro_name: str,
        skip_logger_removal: bool = False,
        custom_provider_registry: ProviderRegistry | None = None,
        provider_data: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.async_client = AsyncLlamaStackAsLibraryClient(
            config_path_or_distro_name, custom_provider_registry, provider_data, skip_logger_removal
        )
        self.provider_data = provider_data

        self.loop = asyncio.new_event_loop()

        # use a new event loop to avoid interfering with the main event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.async_client.initialize())
        finally:
            asyncio.set_event_loop(None)

    def initialize(self):
        """
        Deprecated method for backward compatibility.
        """
        pass

    def shutdown(self) -> None:
        """Shutdown the client and release all resources.

        This method should be called when you're done using the client to properly
        close database connections and release other resources. Failure to call this
        method may result in the program hanging on exit while waiting for background
        threads to complete.

        This method is idempotent and can be called multiple times safely.

        Example:
            client = LlamaStackAsLibraryClient("starter")
            # ... use the client ...
            client.shutdown()
        """
        loop = self.loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.async_client.shutdown())
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def __enter__(self) -> "LlamaStackAsLibraryClient":
        """Enter the context manager.

        The client is already initialized in __init__, so this just returns self.

        Example:
            with LlamaStackAsLibraryClient("starter") as client:
                response = client.models.list()
            # Client is automatically shut down here
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager and shut down the client."""
        self.shutdown()

    def request(self, *args, **kwargs):
        loop = self.loop
        asyncio.set_event_loop(loop)

        if kwargs.get("stream"):

            def sync_generator():
                try:
                    async_stream = loop.run_until_complete(self.async_client.request(*args, **kwargs))
                    while True:
                        chunk = loop.run_until_complete(async_stream.__anext__())
                        yield chunk
                except StopAsyncIteration:
                    pass
                finally:
                    pending = asyncio.all_tasks(loop)
                    if pending:
                        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

            return sync_generator()
        else:
            try:
                result = loop.run_until_complete(self.async_client.request(*args, **kwargs))
            finally:
                pending = asyncio.all_tasks(loop)
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            return result


class AsyncLlamaStackAsLibraryClient(AsyncLlamaStackClient):
    def __init__(
        self,
        config_path_or_distro_name: str,
        custom_provider_registry: ProviderRegistry | None = None,
        provider_data: dict[str, Any] | None = None,
        skip_logger_removal: bool = False,
    ):
        super().__init__()
        # Initialize logging from environment variables first
        setup_logging()
        if in_notebook():
            import nest_asyncio

            nest_asyncio.apply()
            if not skip_logger_removal:
                self._remove_root_logger_handlers()

        if config_path_or_distro_name.endswith(".yaml"):
            config_path = Path(config_path_or_distro_name)
            if not config_path.exists():
                raise ValueError(f"Config file {config_path} does not exist")
            config_dict = replace_env_vars(yaml.safe_load(config_path.read_text()))
            config = parse_and_maybe_upgrade_config(config_dict)
        else:
            # distribution
            config = get_stack_run_config_from_distro(config_path_or_distro_name)

        self.config_path_or_distro_name = config_path_or_distro_name
        self.config = config
        self.custom_provider_registry = custom_provider_registry
        self.provider_data = provider_data
        self.route_impls: RouteImpls | None = None  # Initialize to None to prevent AttributeError
        self.stack: Stack | None = None

    def _remove_root_logger_handlers(self):
        """
        Remove all handlers from the root logger. Needed to avoid polluting the console with logs.
        """
        root_logger = logging.getLogger()

        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            logger.info(f"Removed handler {handler.__class__.__name__} from root logger")

    async def initialize(self) -> bool:
        """
        Initialize the async client.

        Returns:
            bool: True if initialization was successful
        """

        try:
            self.route_impls = None

            self.stack = Stack(self.config, self.custom_provider_registry)
            await self.stack.initialize()
            self.impls = self.stack.impls
        except ModuleNotFoundError as _e:
            cprint(_e.msg, color="red", file=sys.stderr)
            cprint(
                "Using llama-stack as a library requires installing dependencies depending on the distribution (providers) you choose.\n",
                color="yellow",
                file=sys.stderr,
            )
            if self.config_path_or_distro_name.endswith(".yaml"):
                print_pip_install_help(self.config)
            else:
                prefix = "!" if in_notebook() else ""
                cprint(
                    f"Please run:\n\n{prefix}llama stack list-deps {self.config_path_or_distro_name} | xargs -L1 uv pip install\n\n",
                    "yellow",
                    file=sys.stderr,
                )
            cprint(
                "Please check your internet connection and try again.",
                "red",
                file=sys.stderr,
            )
            raise _e

        assert self.impls is not None

        if not os.environ.get("PYTEST_CURRENT_TEST"):
            console = Console()
            console.print(f"Using config [blue]{self.config_path_or_distro_name}[/blue]:")
            safe_config = redact_sensitive_fields(self.config.model_dump())
            console.print(yaml.dump(safe_config, indent=2))

        self.route_impls = initialize_route_impls(self.impls)
        return True

    async def shutdown(self) -> None:
        """Shutdown the client and release all resources.

        This method should be called when you're done using the client to properly
        close database connections and release other resources. Failure to call this
        method may result in the program hanging on exit while waiting for background
        threads to complete.

        This method is idempotent and can be called multiple times safely.

        Example:
            client = AsyncLlamaStackAsLibraryClient("starter")
            await client.initialize()
            # ... use the client ...
            await client.shutdown()
        """
        if self.stack:
            await self.stack.shutdown()
            self.stack = None

    async def __aenter__(self) -> "AsyncLlamaStackAsLibraryClient":
        """Enter the async context manager.

        Initializes the client and returns it.

        Example:
            async with AsyncLlamaStackAsLibraryClient("starter") as client:
                response = await client.models.list()
            # Client is automatically shut down here
        """
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the async context manager and shut down the client."""
        await self.shutdown()

    async def request(
        self,
        cast_to: Any,
        options: Any,
        *,
        stream=False,
        stream_cls=None,
    ):
        if self.route_impls is None:
            raise ValueError("Client not initialized. Please call initialize() first.")

        # Create headers with provider data if available
        headers = options.headers or {}
        if self.provider_data:
            keys = ["X-LlamaStack-Provider-Data", "x-llamastack-provider-data"]
            if all(key not in headers for key in keys):
                headers["X-LlamaStack-Provider-Data"] = json.dumps(self.provider_data)

        # Use context manager for provider data
        with request_provider_data_context(headers):
            if stream:
                response = await self._call_streaming(
                    cast_to=cast_to,
                    options=options,
                    stream_cls=stream_cls,
                )
            else:
                response = await self._call_non_streaming(
                    cast_to=cast_to,
                    options=options,
                )
            return response

    def _handle_file_uploads(self, options: Any, body: dict) -> tuple[dict, list[str]]:
        """Handle file uploads from OpenAI client and add them to the request body."""
        if not (hasattr(options, "files") and options.files):
            return body, []

        if not isinstance(options.files, list):
            return body, []

        field_names = []
        for file_tuple in options.files:
            if not (isinstance(file_tuple, tuple) and len(file_tuple) >= 2):
                continue

            field_name = file_tuple[0]
            file_object = file_tuple[1]

            if isinstance(file_object, BytesIO):
                file_object.seek(0)
                file_content = file_object.read()
                filename = getattr(file_object, "name", "uploaded_file")
                field_names.append(field_name)
                body[field_name] = LibraryClientUploadFile(filename, file_content)

        return body, field_names

    async def _call_non_streaming(
        self,
        *,
        cast_to: Any,
        options: Any,
    ):
        assert self.route_impls is not None  # Should be guaranteed by request() method, assertion for mypy
        path = options.url
        body = options.params or {}
        body |= options.json_data or {}

        # Merge extra_json parameters (extra_body from SDK is converted to extra_json)
        if hasattr(options, "extra_json") and options.extra_json:
            body |= options.extra_json

        matched_func, path_params, route_path, webmethod = find_matching_route(options.method, path, self.route_impls)
        body |= path_params

        # Pass through params that aren't already handled as path params
        if options.params:
            extra_query_params = {k: v for k, v in options.params.items() if k not in path_params}
            if extra_query_params:
                body["extra_query"] = extra_query_params

        body, field_names = self._handle_file_uploads(options, body)

        body = self._convert_body(matched_func, body, exclude_params=set(field_names))
        result = await matched_func(**body)

        # Handle FastAPI Response objects (e.g., from file content retrieval)
        if isinstance(result, FastAPIResponse):
            return LibraryClientHttpxResponse(result)

        json_content = json.dumps(convert_pydantic_to_json_value(result))

        filtered_body = {k: v for k, v in body.items() if not isinstance(v, LibraryClientUploadFile)}

        status_code = httpx.codes.OK

        if options.method.upper() == "DELETE" and result is None:
            status_code = httpx.codes.NO_CONTENT

        if status_code == httpx.codes.NO_CONTENT:
            json_content = ""

        mock_response = httpx.Response(
            status_code=status_code,
            content=json_content.encode("utf-8"),
            headers={
                "Content-Type": "application/json",
            },
            request=httpx.Request(
                method=options.method,
                url=options.url,
                params=options.params,
                headers=options.headers or {},
                json=convert_pydantic_to_json_value(filtered_body),
            ),
        )
        response = APIResponse(
            raw=mock_response,
            client=self,
            cast_to=cast_to,
            options=options,
            stream=False,
            stream_cls=None,
        )
        return response.parse()

    async def _call_streaming(
        self,
        *,
        cast_to: Any,
        options: Any,
        stream_cls: Any,
    ):
        assert self.route_impls is not None  # Should be guaranteed by request() method, assertion for mypy
        path = options.url
        body = options.params or {}
        body |= options.json_data or {}
        func, path_params, route_path, webmethod = find_matching_route(options.method, path, self.route_impls)
        body |= path_params

        # Prepare body for the function call (handles both Pydantic and traditional params)
        body = self._convert_body(func, body)

        async def gen():
            async for chunk in await func(**body):
                data = json.dumps(convert_pydantic_to_json_value(chunk))
                sse_event = f"data: {data}\n\n"
                yield sse_event.encode("utf-8")

        wrapped_gen = preserve_contexts_async_generator(gen(), [PROVIDER_DATA_VAR])

        mock_response = httpx.Response(
            status_code=httpx.codes.OK,
            content=wrapped_gen,
            headers={
                "Content-Type": "application/json",
            },
            request=httpx.Request(
                method=options.method,
                url=options.url,
                params=options.params,
                headers=options.headers or {},
                json=convert_pydantic_to_json_value(body),
            ),
        )

        # we use asynchronous impl always internally and channel all requests to AsyncLlamaStackClient
        # however, the top-level caller may be a SyncAPIClient -- so its stream_cls might be a Stream (SyncStream)
        # so we need to convert it to AsyncStream
        # mypy can't track runtime variables inside the [...] of a generic, so ignore that check
        args = get_args(stream_cls)
        stream_cls = AsyncStream[args[0]]  # type: ignore[valid-type]
        response = AsyncAPIResponse(
            raw=mock_response,
            client=self,
            cast_to=cast_to,
            options=options,
            stream=True,
            stream_cls=stream_cls,
        )
        return await response.parse()

    def _convert_body(self, func: Any, body: dict | None = None, exclude_params: set[str] | None = None) -> dict:
        body = body or {}
        exclude_params = exclude_params or set()
        sig = inspect.signature(func)
        params_list = [p for p in sig.parameters.values() if p.name != "self"]

        # Flatten if there's a single unwrapped body parameter (BaseModel or Annotated[BaseModel, Body(embed=False)])
        if len(params_list) == 1:
            param = params_list[0]
            param_type = param.annotation
            if is_unwrapped_body_param(param_type):
                base_type = get_args(param_type)[0]
                return {param.name: base_type(**body)}

        # Strip NOT_GIVENs to use the defaults in signature
        body = {k: v for k, v in body.items() if v is not NOT_GIVEN}

        # Check if there's an unwrapped body parameter among multiple parameters
        # (e.g., path param + body param like: vector_store_id: str, params: Annotated[Model, Body(...)])
        unwrapped_body_param = None
        for param in params_list:
            if is_unwrapped_body_param(param.annotation):
                unwrapped_body_param = param
                break

        # Check for parameters with Depends() annotation (FastAPI router endpoints)
        # These need special handling: construct the request model from body
        depends_param = None
        for param in params_list:
            param_type = param.annotation
            if get_origin(param_type) is typing.Annotated:
                args = get_args(param_type)
                if len(args) > 1:
                    # Check if any metadata is Depends
                    metadata = args[1:]
                    for item in metadata:
                        # Check if it's a Depends object (has dependency attribute or is a callable)
                        # Depends objects typically have a 'dependency' attribute or are callable functions
                        if hasattr(item, "dependency") or callable(item) or "Depends" in str(type(item)):
                            depends_param = param
                            break
                if depends_param:
                    break

        # Convert parameters to Pydantic models where needed
        converted_body = {}
        for param_name, param in sig.parameters.items():
            if param_name in body:
                value = body.get(param_name)
                if param_name in exclude_params:
                    converted_body[param_name] = value
                else:
                    converted_body[param_name] = convert_to_pydantic(param.annotation, value)

        # Handle Depends parameter: construct request model from body
        if depends_param and depends_param.name not in converted_body:
            param_type = depends_param.annotation
            if get_origin(param_type) is typing.Annotated:
                base_type = get_args(param_type)[0]
                # Handle Union types (e.g., SomeRequestModel | None) - extract the non-None type
                # In Python 3.10+, Union types created with | syntax are still typing.Union
                origin = get_origin(base_type)
                if origin is Union:
                    # Get the first non-None type from the Union
                    union_args = get_args(base_type)
                    base_type = next(
                        (t for t in union_args if t is not type(None) and t is not None),
                        union_args[0] if union_args else None,
                    )

                # Only try to instantiate if it's a class (not a Union or other non-callable type)
                if base_type is not None and inspect.isclass(base_type) and callable(base_type):
                    # Construct the request model from all body parameters
                    converted_body[depends_param.name] = base_type(**body)

        # handle unwrapped body parameter after processing all named parameters
        if unwrapped_body_param:
            base_type = get_args(unwrapped_body_param.annotation)[0]
            # extract only keys not already used by other params
            remaining_keys = {k: v for k, v in body.items() if k not in converted_body}
            converted_body[unwrapped_body_param.name] = base_type(**remaining_keys)

        return converted_body

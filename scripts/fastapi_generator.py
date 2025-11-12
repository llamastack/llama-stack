#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
FastAPI-based OpenAPI generator for Llama Stack.
"""

import importlib
import inspect
import pkgutil
import types
import typing
from pathlib import Path
from typing import Annotated, Any, get_args, get_origin

import yaml
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from openapi_spec_validator import validate_spec
from openapi_spec_validator.exceptions import OpenAPISpecValidatorError

from llama_stack.apis.datatypes import Api
from llama_stack.apis.version import (
    LLAMA_STACK_API_V1,
    LLAMA_STACK_API_V1ALPHA,
    LLAMA_STACK_API_V1BETA,
)
from llama_stack.core.resolver import api_protocol_map

# Global list to store dynamic models created during endpoint generation
_dynamic_models = []


# Cache for protocol methods to avoid repeated lookups
_protocol_methods_cache: dict[Api, dict[str, Any]] | None = None

# Global dict to store extra body field information by endpoint
# Key: (path, method) tuple, Value: list of (param_name, param_type, description) tuples
_extra_body_fields: dict[tuple[str, str], list[tuple[str, type, str | None]]] = {}


def create_llama_stack_app() -> FastAPI:
    """
    Create a FastAPI app that represents the Llama Stack API.
    This uses the existing route discovery system to automatically find all routes.
    """
    app = FastAPI(
        title="Llama Stack API",
        description="A comprehensive API for building and deploying AI applications",
        version="1.0.0",
        servers=[
            {"url": "http://any-hosted-llama-stack.com"},
        ],
    )

    # Get all API routes
    from llama_stack.core.server.routes import get_all_api_routes

    api_routes = get_all_api_routes()

    # Create FastAPI routes from the discovered routes
    for api, routes in api_routes.items():
        for route, webmethod in routes:
            # Convert the route to a FastAPI endpoint
            _create_fastapi_endpoint(app, route, webmethod, api)

    return app


def _get_protocol_method(api: Api, method_name: str) -> Any | None:
    """
    Get a protocol method function by API and method name.
    Uses caching to avoid repeated lookups.

    Args:
        api: The API enum
        method_name: The method name (function name)

    Returns:
        The function object, or None if not found
    """
    global _protocol_methods_cache

    if _protocol_methods_cache is None:
        _protocol_methods_cache = {}
        protocols = api_protocol_map()
        from llama_stack.apis.tools import SpecialToolGroup, ToolRuntime

        toolgroup_protocols = {
            SpecialToolGroup.rag_tool: ToolRuntime,
        }

        for api_key, protocol in protocols.items():
            method_map: dict[str, Any] = {}
            protocol_methods = inspect.getmembers(protocol, predicate=inspect.isfunction)
            for name, method in protocol_methods:
                method_map[name] = method

            # Handle tool_runtime special case
            if api_key == Api.tool_runtime:
                for tool_group, sub_protocol in toolgroup_protocols.items():
                    sub_protocol_methods = inspect.getmembers(sub_protocol, predicate=inspect.isfunction)
                    for name, method in sub_protocol_methods:
                        if hasattr(method, "__webmethod__"):
                            method_map[f"{tool_group.value}.{name}"] = method

            _protocol_methods_cache[api_key] = method_map

    return _protocol_methods_cache.get(api, {}).get(method_name)


def _extract_path_parameters(path: str) -> list[dict[str, Any]]:
    """Extract path parameters from a URL path and return them as OpenAPI parameter definitions."""
    import re

    matches = re.findall(r"\{([^}:]+)(?::[^}]+)?\}", path)
    return [
        {
            "name": param_name,
            "in": "path",
            "required": True,
            "schema": {"type": "string"},
            "description": f"Path parameter: {param_name}",
        }
        for param_name in matches
    ]


def _create_endpoint_with_request_model(
    request_model: type, response_model: type | None, operation_description: str | None
):
    """Create an endpoint function with a request body model."""

    async def endpoint(request: request_model) -> response_model:
        return response_model() if response_model else {}

    if operation_description:
        endpoint.__doc__ = operation_description
    return endpoint


def _build_field_definitions(query_parameters: list[tuple[str, type, Any]], use_any: bool = False) -> dict[str, tuple]:
    """Build field definitions for a Pydantic model from query parameters."""
    from typing import Any

    from pydantic import Field

    field_definitions = {}
    for param_name, param_type, default_value in query_parameters:
        if use_any:
            field_definitions[param_name] = (Any, ... if default_value is inspect.Parameter.empty else default_value)
            continue

        base_type = param_type
        extracted_field = None
        if get_origin(param_type) is Annotated:
            args = get_args(param_type)
            if args:
                base_type = args[0]
                for arg in args[1:]:
                    if isinstance(arg, Field):
                        extracted_field = arg
                        break

        try:
            if extracted_field:
                field_definitions[param_name] = (base_type, extracted_field)
            else:
                field_definitions[param_name] = (
                    base_type,
                    ... if default_value is inspect.Parameter.empty else default_value,
                )
        except (TypeError, ValueError):
            field_definitions[param_name] = (Any, ... if default_value is inspect.Parameter.empty else default_value)

    # Ensure all parameters are included
    expected_params = {name for name, _, _ in query_parameters}
    missing = expected_params - set(field_definitions.keys())
    if missing:
        for param_name, _, default_value in query_parameters:
            if param_name in missing:
                field_definitions[param_name] = (
                    Any,
                    ... if default_value is inspect.Parameter.empty else default_value,
                )

    return field_definitions


def _create_dynamic_request_model(
    webmethod, query_parameters: list[tuple[str, type, Any]], use_any: bool = False, add_uuid: bool = False
) -> type | None:
    """Create a dynamic Pydantic model for request body."""
    import uuid

    from pydantic import create_model

    try:
        field_definitions = _build_field_definitions(query_parameters, use_any)
        if not field_definitions:
            return None
        clean_route = webmethod.route.replace("/", "_").replace("{", "").replace("}", "").replace("-", "_")
        model_name = f"{clean_route}_Request"
        if add_uuid:
            model_name = f"{model_name}_{uuid.uuid4().hex[:8]}"

        request_model = create_model(model_name, **field_definitions)
        _dynamic_models.append(request_model)
        return request_model
    except Exception:
        return None


def _build_signature_params(
    query_parameters: list[tuple[str, type, Any]],
) -> tuple[list[inspect.Parameter], dict[str, type]]:
    """Build signature parameters and annotations from query parameters."""
    signature_params = []
    param_annotations = {}
    for param_name, param_type, default_value in query_parameters:
        param_annotations[param_name] = param_type
        signature_params.append(
            inspect.Parameter(
                param_name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default_value if default_value is not inspect.Parameter.empty else inspect.Parameter.empty,
                annotation=param_type,
            )
        )
    return signature_params, param_annotations


def _create_fastapi_endpoint(app: FastAPI, route, webmethod, api: Api):
    """Create a FastAPI endpoint from a discovered route and webmethod."""
    path = route.path
    methods = route.methods
    name = route.name
    fastapi_path = path.replace("{", "{").replace("}", "}")
    is_post_put = any(method.upper() in ["POST", "PUT", "PATCH"] for method in methods)

    request_model, response_model, query_parameters, file_form_params, streaming_response_model = (
        _find_models_for_endpoint(webmethod, api, name, is_post_put)
    )
    operation_description = _extract_operation_description_from_docstring(api, name)
    response_description = _extract_response_description_from_docstring(webmethod, response_model, api, name)

    # Retrieve and store extra body fields for this endpoint
    func = _get_protocol_method(api, name)
    extra_body_params = getattr(func, "_extra_body_params", []) if func else []
    if extra_body_params:
        global _extra_body_fields
        for method in methods:
            key = (fastapi_path, method.upper())
            _extra_body_fields[key] = extra_body_params

    if file_form_params and is_post_put:
        signature_params = list(file_form_params)
        param_annotations = {param.name: param.annotation for param in file_form_params}
        for param_name, param_type, default_value in query_parameters:
            signature_params.append(
                inspect.Parameter(
                    param_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=default_value if default_value is not inspect.Parameter.empty else inspect.Parameter.empty,
                    annotation=param_type,
                )
            )
            param_annotations[param_name] = param_type

        async def file_form_endpoint():
            return response_model() if response_model else {}

        if operation_description:
            file_form_endpoint.__doc__ = operation_description
        file_form_endpoint.__signature__ = inspect.Signature(signature_params)
        file_form_endpoint.__annotations__ = param_annotations
        endpoint_func = file_form_endpoint
    elif request_model and response_model:
        endpoint_func = _create_endpoint_with_request_model(request_model, response_model, operation_description)
    elif response_model and query_parameters:
        if is_post_put:
            # Try creating request model with type preservation, fallback to Any, then minimal
            request_model = _create_dynamic_request_model(webmethod, query_parameters, use_any=False)
            if not request_model:
                request_model = _create_dynamic_request_model(webmethod, query_parameters, use_any=True)
            if not request_model:
                request_model = _create_dynamic_request_model(webmethod, query_parameters, use_any=True, add_uuid=True)

            if request_model:
                endpoint_func = _create_endpoint_with_request_model(
                    request_model, response_model, operation_description
                )
            else:

                async def empty_endpoint() -> response_model:
                    return response_model() if response_model else {}

                if operation_description:
                    empty_endpoint.__doc__ = operation_description
                endpoint_func = empty_endpoint
        else:
            sorted_params = sorted(query_parameters, key=lambda x: (x[2] is not inspect.Parameter.empty, x[0]))
            signature_params, param_annotations = _build_signature_params(sorted_params)

            async def query_endpoint():
                return response_model()

            if operation_description:
                query_endpoint.__doc__ = operation_description
            query_endpoint.__signature__ = inspect.Signature(signature_params)
            query_endpoint.__annotations__ = param_annotations
            endpoint_func = query_endpoint
    elif response_model:

        async def response_only_endpoint() -> response_model:
            return response_model()

        if operation_description:
            response_only_endpoint.__doc__ = operation_description
        endpoint_func = response_only_endpoint
    elif query_parameters:
        signature_params, param_annotations = _build_signature_params(query_parameters)

        async def params_only_endpoint():
            return {}

        if operation_description:
            params_only_endpoint.__doc__ = operation_description
        params_only_endpoint.__signature__ = inspect.Signature(signature_params)
        params_only_endpoint.__annotations__ = param_annotations
        endpoint_func = params_only_endpoint
    else:
        # Endpoint with no parameters and no response model
        # If we have a response_model from the function signature, use it even if _find_models_for_endpoint didn't find it
        # This can happen if there was an exception during model finding
        if response_model is None:
            # Try to get response model directly from the function signature as a fallback
            func = _get_protocol_method(api, name)
            if func:
                try:
                    sig = inspect.signature(func)
                    return_annotation = sig.return_annotation
                    if return_annotation != inspect.Signature.empty:
                        if hasattr(return_annotation, "model_json_schema"):
                            response_model = return_annotation
                        elif get_origin(return_annotation) is Annotated:
                            args = get_args(return_annotation)
                            if args and hasattr(args[0], "model_json_schema"):
                                response_model = args[0]
                except Exception:
                    pass

        if response_model:

            async def no_params_endpoint() -> response_model:
                return response_model() if response_model else {}
        else:

            async def no_params_endpoint():
                return {}

        if operation_description:
            no_params_endpoint.__doc__ = operation_description
        endpoint_func = no_params_endpoint

    # Build response content with both application/json and text/event-stream if streaming
    response_content = {}
    if response_model:
        response_content["application/json"] = {"schema": {"$ref": f"#/components/schemas/{response_model.__name__}"}}
    if streaming_response_model:
        # Get the schema name for the streaming model
        # It might be a registered schema or a Pydantic model
        streaming_schema_name = None
        # Check if it's a registered schema first (before checking __name__)
        # because registered schemas might be Annotated types
        from llama_stack.schema_utils import _registered_schemas

        if streaming_response_model in _registered_schemas:
            streaming_schema_name = _registered_schemas[streaming_response_model]["name"]
        elif hasattr(streaming_response_model, "__name__"):
            streaming_schema_name = streaming_response_model.__name__

        if streaming_schema_name:
            response_content["text/event-stream"] = {
                "schema": {"$ref": f"#/components/schemas/{streaming_schema_name}"}
            }

    # If no content types, use empty schema
    if not response_content:
        response_content["application/json"] = {"schema": {}}

    # Add the endpoint to the FastAPI app
    is_deprecated = webmethod.deprecated or False
    route_kwargs = {
        "name": name,
        "tags": [_get_tag_from_api(api)],
        "deprecated": is_deprecated,
        "responses": {
            200: {
                "description": response_description,
                "content": response_content,
            },
            400: {"$ref": "#/components/responses/BadRequest400"},
            429: {"$ref": "#/components/responses/TooManyRequests429"},
            500: {"$ref": "#/components/responses/InternalServerError500"},
            "default": {"$ref": "#/components/responses/DefaultError"},
        },
    }

    # FastAPI needs response_model parameter to properly generate OpenAPI spec
    # Use the non-streaming response model if available
    if response_model:
        route_kwargs["response_model"] = response_model

    method_map = {"GET": app.get, "POST": app.post, "PUT": app.put, "DELETE": app.delete, "PATCH": app.patch}
    for method in methods:
        if handler := method_map.get(method.upper()):
            handler(fastapi_path, **route_kwargs)(endpoint_func)


def _extract_operation_description_from_docstring(api: Api, method_name: str) -> str | None:
    """Extract operation description from the actual function docstring."""
    func = _get_protocol_method(api, method_name)
    if not func or not func.__doc__:
        return None

    doc_lines = func.__doc__.split("\n")
    description_lines = []
    metadata_markers = (":param", ":type", ":return", ":returns", ":raises", ":exception", ":yield", ":yields", ":cvar")

    for line in doc_lines:
        if line.strip().startswith(metadata_markers):
            break
        description_lines.append(line)

    description = "\n".join(description_lines).strip()
    return description if description else None


def _extract_response_description_from_docstring(webmethod, response_model, api: Api, method_name: str) -> str:
    """Extract response description from the actual function docstring."""
    func = _get_protocol_method(api, method_name)
    if not func or not func.__doc__:
        return "Successful Response"
    for line in func.__doc__.split("\n"):
        if line.strip().startswith(":returns:"):
            if desc := line.strip()[9:].strip():
                return desc
    return "Successful Response"


def _get_tag_from_api(api: Api) -> str:
    """Extract a tag name from the API enum for API grouping."""
    return api.value.replace("_", " ").title()


def _is_file_or_form_param(param_type: Any) -> bool:
    """Check if a parameter type is annotated with File() or Form()."""
    if get_origin(param_type) is Annotated:
        args = get_args(param_type)
        if len(args) > 1:
            # Check metadata for File or Form
            for metadata in args[1:]:
                # Check if it's a File or Form instance
                if hasattr(metadata, "__class__"):
                    class_name = metadata.__class__.__name__
                    if class_name in ("File", "Form"):
                        return True
    return False


def _is_extra_body_field(metadata_item: Any) -> bool:
    """Check if a metadata item is an ExtraBodyField instance."""
    from llama_stack.schema_utils import ExtraBodyField

    return isinstance(metadata_item, ExtraBodyField)


def _is_async_iterator_type(type_obj: Any) -> bool:
    """Check if a type is AsyncIterator or AsyncIterable."""
    from collections.abc import AsyncIterable, AsyncIterator

    origin = get_origin(type_obj)
    if origin is None:
        # Check if it's the class itself
        return type_obj in (AsyncIterator, AsyncIterable) or (
            hasattr(type_obj, "__origin__") and type_obj.__origin__ in (AsyncIterator, AsyncIterable)
        )
    return origin in (AsyncIterator, AsyncIterable)


def _extract_response_models_from_union(union_type: Any) -> tuple[type | None, type | None]:
    """
    Extract non-streaming and streaming response models from a union type.

    Returns:
        tuple: (non_streaming_model, streaming_model)
    """
    non_streaming_model = None
    streaming_model = None

    args = get_args(union_type)
    for arg in args:
        # Check if it's an AsyncIterator
        if _is_async_iterator_type(arg):
            # Extract the type argument from AsyncIterator[T]
            iterator_args = get_args(arg)
            if iterator_args:
                inner_type = iterator_args[0]
                # Check if the inner type is a registered schema (union type)
                # or a Pydantic model
                if hasattr(inner_type, "model_json_schema"):
                    streaming_model = inner_type
                else:
                    # Might be a registered schema - check if it's registered
                    from llama_stack.schema_utils import _registered_schemas

                    if inner_type in _registered_schemas:
                        # We'll need to look this up later, but for now store the type
                        streaming_model = inner_type
        elif hasattr(arg, "model_json_schema"):
            # Non-streaming Pydantic model
            if non_streaming_model is None:
                non_streaming_model = arg

    return non_streaming_model, streaming_model


def _find_models_for_endpoint(
    webmethod, api: Api, method_name: str, is_post_put: bool = False
) -> tuple[type | None, type | None, list[tuple[str, type, Any]], list[inspect.Parameter], type | None]:
    """
    Find appropriate request and response models for an endpoint by analyzing the actual function signature.
    This uses the protocol function to determine the correct models dynamically.

    Args:
        webmethod: The webmethod metadata
        api: The API enum for looking up the function
        method_name: The method name (function name)
        is_post_put: Whether this is a POST, PUT, or PATCH request (GET requests should never have request bodies)

    Returns:
        tuple: (request_model, response_model, query_parameters, file_form_params, streaming_response_model)
        where query_parameters is a list of (name, type, default_value) tuples
        and file_form_params is a list of inspect.Parameter objects for File()/Form() params
        and streaming_response_model is the model for streaming responses (AsyncIterator content)
    """
    try:
        # Get the function from the protocol
        func = _get_protocol_method(api, method_name)
        if not func:
            return None, None, [], [], None

        # Analyze the function signature
        sig = inspect.signature(func)

        # Find request model and collect all body parameters
        request_model = None
        query_parameters = []
        file_form_params = []
        path_params = set()
        extra_body_params = []

        # Extract path parameters from the route
        if webmethod and hasattr(webmethod, "route"):
            import re

            path_matches = re.findall(r"\{([^}:]+)(?::[^}]+)?\}", webmethod.route)
            path_params = set(path_matches)

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            # Skip *args and **kwargs parameters - these are not real API parameters
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue

            # Check if this is a path parameter
            if param_name in path_params:
                # Path parameters are handled separately, skip them
                continue

            # Check if it's a File() or Form() parameter - these need special handling
            param_type = param.annotation
            if _is_file_or_form_param(param_type):
                # File() and Form() parameters must be in the function signature directly
                # They cannot be part of a Pydantic model
                file_form_params.append(param)
                continue

            # Check for ExtraBodyField in Annotated types
            is_extra_body = False
            extra_body_description = None
            if get_origin(param_type) is Annotated:
                args = get_args(param_type)
                base_type = args[0] if args else param_type
                metadata = args[1:] if len(args) > 1 else []

                # Check if any metadata item is an ExtraBodyField
                for metadata_item in metadata:
                    if _is_extra_body_field(metadata_item):
                        is_extra_body = True
                        extra_body_description = metadata_item.description
                        break

                if is_extra_body:
                    # Store as extra body parameter - exclude from request model
                    extra_body_params.append((param_name, base_type, extra_body_description))
                    continue

            # Check if it's a Pydantic model (for POST/PUT requests)
            if hasattr(param_type, "model_json_schema"):
                # Collect all body parameters including Pydantic models
                # We'll decide later whether to use a single model or create a combined one
                query_parameters.append((param_name, param_type, param.default))
            elif get_origin(param_type) is Annotated:
                # Handle Annotated types - get the base type
                args = get_args(param_type)
                if args and hasattr(args[0], "model_json_schema"):
                    # Collect Pydantic models from Annotated types
                    query_parameters.append((param_name, args[0], param.default))
                else:
                    # Regular annotated parameter (but not File/Form, already handled above)
                    query_parameters.append((param_name, param_type, param.default))
            else:
                # This is likely a body parameter for POST/PUT or query parameter for GET
                # Store the parameter info for later use
                # Preserve inspect.Parameter.empty to distinguish "no default" from "default=None"
                default_value = param.default

                # Extract the base type from union types (e.g., str | None -> str)
                # Also make it safe for FastAPI to avoid forward reference issues
                query_parameters.append((param_name, param_type, default_value))

        # Store extra body fields for later use in post-processing
        # We'll store them when the endpoint is created, as we need the full path
        # For now, attach to the function for later retrieval
        if extra_body_params:
            func._extra_body_params = extra_body_params  # type: ignore

        # If there's exactly one body parameter and it's a Pydantic model, use it directly
        # Otherwise, we'll create a combined request model from all parameters
        # BUT: For GET requests, never create a request body - all parameters should be query parameters
        if is_post_put and len(query_parameters) == 1:
            param_name, param_type, default_value = query_parameters[0]
            if hasattr(param_type, "model_json_schema"):
                request_model = param_type
                query_parameters = []  # Clear query_parameters so we use the single model

        # Find response model from return annotation
        # Also detect streaming response models (AsyncIterator)
        response_model = None
        streaming_response_model = None
        return_annotation = sig.return_annotation
        if return_annotation != inspect.Signature.empty:
            origin = get_origin(return_annotation)
            if hasattr(return_annotation, "model_json_schema"):
                response_model = return_annotation
            elif origin is Annotated:
                # Handle Annotated return types
                args = get_args(return_annotation)
                if args:
                    # Check if the first argument is a Pydantic model
                    if hasattr(args[0], "model_json_schema"):
                        response_model = args[0]
                    else:
                        # Check if the first argument is a union type
                        inner_origin = get_origin(args[0])
                        if inner_origin is not None and (
                            inner_origin is types.UnionType or inner_origin is typing.Union
                        ):
                            response_model, streaming_response_model = _extract_response_models_from_union(args[0])
            elif origin is not None and (origin is types.UnionType or origin is typing.Union):
                # Handle union types - extract both non-streaming and streaming models
                response_model, streaming_response_model = _extract_response_models_from_union(return_annotation)

        return request_model, response_model, query_parameters, file_form_params, streaming_response_model

    except Exception:
        # If we can't analyze the function signature, return None
        return None, None, [], [], None


def _ensure_components_schemas(openapi_schema: dict[str, Any]) -> None:
    """Ensure components.schemas exists in the schema."""
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
    if "schemas" not in openapi_schema["components"]:
        openapi_schema["components"]["schemas"] = {}


def _import_all_modules_in_package(package_name: str) -> list[Any]:
    """
    Dynamically import all modules in a package to trigger register_schema calls.

    This walks through all modules in the package and imports them, ensuring
    that any register_schema() calls at module level are executed.

    Args:
        package_name: The fully qualified package name (e.g., 'llama_stack.apis')

    Returns:
        List of imported module objects
    """
    modules = []
    try:
        package = importlib.import_module(package_name)
    except ImportError:
        return modules

    package_path = getattr(package, "__path__", None)
    if not package_path:
        return modules

    # Walk packages and modules recursively
    for _, modname, ispkg in pkgutil.walk_packages(package_path, prefix=f"{package_name}."):
        if not modname.startswith("_"):
            try:
                module = importlib.import_module(modname)
                modules.append(module)

                # If this is a package, also try to import any .py files directly
                # (e.g., llama_stack.apis.scoring_functions.scoring_functions)
                if ispkg:
                    try:
                        # Try importing the module file with the same name as the package
                        # This handles cases like scoring_functions/scoring_functions.py
                        module_file_name = f"{modname}.{modname.split('.')[-1]}"
                        module_file = importlib.import_module(module_file_name)
                        if module_file not in modules:
                            modules.append(module_file)
                    except (ImportError, AttributeError, TypeError):
                        # It's okay if this fails - not all packages have a module file with the same name
                        pass
            except (ImportError, AttributeError, TypeError):
                # Skip modules that can't be imported (e.g., missing dependencies)
                continue

    return modules


def _extract_and_fix_defs(schema: dict[str, Any], openapi_schema: dict[str, Any]) -> None:
    """
    Extract $defs from a schema, move them to components/schemas, and fix references.
    This handles both TypeAdapter-generated schemas and model_json_schema() schemas.
    """
    if "$defs" in schema:
        defs = schema.pop("$defs")
        for def_name, def_schema in defs.items():
            if def_name not in openapi_schema["components"]["schemas"]:
                openapi_schema["components"]["schemas"][def_name] = def_schema
                # Recursively handle $defs in nested schemas
                _extract_and_fix_defs(def_schema, openapi_schema)

        # Fix any references in the main schema that point to $defs
        def fix_refs_in_schema(obj: Any) -> None:
            if isinstance(obj, dict):
                if "$ref" in obj and obj["$ref"].startswith("#/$defs/"):
                    obj["$ref"] = obj["$ref"].replace("#/$defs/", "#/components/schemas/")
                for value in obj.values():
                    fix_refs_in_schema(value)
            elif isinstance(obj, list):
                for item in obj:
                    fix_refs_in_schema(item)

        fix_refs_in_schema(schema)


def _ensure_json_schema_types_included(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Ensure all @json_schema_type decorated models and registered schemas are included in the OpenAPI schema.
    This finds all models with the _llama_stack_schema_type attribute and schemas registered via register_schema.
    """
    _ensure_components_schemas(openapi_schema)

    # Import TypeAdapter for handling union types and other non-model types
    from pydantic import TypeAdapter

    # Dynamically import all modules in packages that might register schemas
    # This ensures register_schema() calls execute and populate _registered_schemas
    # Also collect the modules for later scanning of @json_schema_type decorated classes
    apis_modules = _import_all_modules_in_package("llama_stack.apis")
    _import_all_modules_in_package("llama_stack.core.telemetry")

    # First, handle registered schemas (union types, etc.)
    from llama_stack.schema_utils import _registered_schemas

    for schema_type, registration_info in _registered_schemas.items():
        schema_name = registration_info["name"]
        if schema_name not in openapi_schema["components"]["schemas"]:
            try:
                # Use TypeAdapter for union types and other non-model types
                # Use ref_template to generate references in the format we need
                adapter = TypeAdapter(schema_type)
                schema = adapter.json_schema(ref_template="#/components/schemas/{model}")

                # Extract and fix $defs if present
                _extract_and_fix_defs(schema, openapi_schema)

                openapi_schema["components"]["schemas"][schema_name] = schema
            except Exception as e:
                # Skip if we can't generate the schema
                print(f"Warning: Failed to generate schema for registered type {schema_name}: {e}")
                import traceback

                traceback.print_exc()
                continue

    # Find all classes with the _llama_stack_schema_type attribute
    # Use the modules we already imported above
    for module in apis_modules:
        for attr_name in dir(module):
            try:
                attr = getattr(module, attr_name)
                if (
                    hasattr(attr, "_llama_stack_schema_type")
                    and hasattr(attr, "model_json_schema")
                    and hasattr(attr, "__name__")
                ):
                    schema_name = attr.__name__
                    if schema_name not in openapi_schema["components"]["schemas"]:
                        try:
                            # Use ref_template to ensure consistent reference format and $defs handling
                            schema = attr.model_json_schema(ref_template="#/components/schemas/{model}")
                            # Extract and fix $defs if present (model_json_schema can also generate $defs)
                            _extract_and_fix_defs(schema, openapi_schema)
                            openapi_schema["components"]["schemas"][schema_name] = schema
                        except Exception as e:
                            # Skip if we can't generate the schema
                            print(f"Warning: Failed to generate schema for {schema_name}: {e}")
                            continue
            except (AttributeError, TypeError):
                continue

    # Also include any dynamic models that were created during endpoint generation
    # This is a workaround to ensure dynamic models appear in the schema
    global _dynamic_models
    if "_dynamic_models" in globals():
        for model in _dynamic_models:
            try:
                schema_name = model.__name__
                if schema_name not in openapi_schema["components"]["schemas"]:
                    schema = model.model_json_schema(ref_template="#/components/schemas/{model}")
                    # Extract and fix $defs if present
                    _extract_and_fix_defs(schema, openapi_schema)
                    openapi_schema["components"]["schemas"][schema_name] = schema
            except Exception:
                # Skip if we can't generate the schema
                continue

    return openapi_schema


def _fix_ref_references(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Fix $ref references to point to components/schemas instead of $defs.
    This prevents the YAML dumper from creating a root-level $defs section.
    """

    def fix_refs(obj: Any) -> None:
        if isinstance(obj, dict):
            if "$ref" in obj and obj["$ref"].startswith("#/$defs/"):
                # Replace #/$defs/ with #/components/schemas/
                obj["$ref"] = obj["$ref"].replace("#/$defs/", "#/components/schemas/")
            for value in obj.values():
                fix_refs(value)
        elif isinstance(obj, list):
            for item in obj:
                fix_refs(item)

    fix_refs(openapi_schema)
    return openapi_schema


def _eliminate_defs_section(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Eliminate $defs section entirely by moving all definitions to components/schemas.
    This matches the structure of the old pyopenapi generator for oasdiff compatibility.
    """
    _ensure_components_schemas(openapi_schema)

    # First pass: collect all $defs from anywhere in the schema
    defs_to_move = {}

    def collect_defs(obj: Any) -> None:
        if isinstance(obj, dict):
            if "$defs" in obj:
                # Collect $defs for later processing
                for def_name, def_schema in obj["$defs"].items():
                    if def_name not in defs_to_move:
                        defs_to_move[def_name] = def_schema

            # Recursively process all values
            for value in obj.values():
                collect_defs(value)
        elif isinstance(obj, list):
            for item in obj:
                collect_defs(item)

    # Collect all $defs
    collect_defs(openapi_schema)

    # Move all $defs to components/schemas
    for def_name, def_schema in defs_to_move.items():
        if def_name not in openapi_schema["components"]["schemas"]:
            openapi_schema["components"]["schemas"][def_name] = def_schema

    # Also move any existing root-level $defs to components/schemas
    if "$defs" in openapi_schema:
        print(f"Found root-level $defs with {len(openapi_schema['$defs'])} items, moving to components/schemas")
        for def_name, def_schema in openapi_schema["$defs"].items():
            if def_name not in openapi_schema["components"]["schemas"]:
                openapi_schema["components"]["schemas"][def_name] = def_schema
        # Remove the root-level $defs
        del openapi_schema["$defs"]

    # Second pass: remove all $defs sections from anywhere in the schema
    def remove_defs(obj: Any) -> None:
        if isinstance(obj, dict):
            if "$defs" in obj:
                del obj["$defs"]

            # Recursively process all values
            for value in obj.values():
                remove_defs(value)
        elif isinstance(obj, list):
            for item in obj:
                remove_defs(item)

    # Remove all $defs sections
    remove_defs(openapi_schema)

    return openapi_schema


def _add_error_responses(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Add standard error response definitions to the OpenAPI schema.
    Uses the actual Error model from the codebase for consistency.
    """
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
    if "responses" not in openapi_schema["components"]:
        openapi_schema["components"]["responses"] = {}

    try:
        from llama_stack.apis.datatypes import Error

        _ensure_components_schemas(openapi_schema)
        if "Error" not in openapi_schema["components"]["schemas"]:
            openapi_schema["components"]["schemas"]["Error"] = Error.model_json_schema()
    except ImportError:
        pass

    # Define standard HTTP error responses
    error_responses = {
        400: {
            "name": "BadRequest400",
            "description": "The request was invalid or malformed",
            "example": {"status": 400, "title": "Bad Request", "detail": "The request was invalid or malformed"},
        },
        429: {
            "name": "TooManyRequests429",
            "description": "The client has sent too many requests in a given amount of time",
            "example": {
                "status": 429,
                "title": "Too Many Requests",
                "detail": "You have exceeded the rate limit. Please try again later.",
            },
        },
        500: {
            "name": "InternalServerError500",
            "description": "The server encountered an unexpected error",
            "example": {"status": 500, "title": "Internal Server Error", "detail": "An unexpected error occurred"},
        },
    }

    # Add each error response to the schema
    for _, error_info in error_responses.items():
        response_name = error_info["name"]
        openapi_schema["components"]["responses"][response_name] = {
            "description": error_info["description"],
            "content": {
                "application/json": {"schema": {"$ref": "#/components/schemas/Error"}, "example": error_info["example"]}
            },
        }

    # Add a default error response
    openapi_schema["components"]["responses"]["DefaultError"] = {
        "description": "An error occurred",
        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}},
    }

    return openapi_schema


def _fix_path_parameters(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Fix path parameter resolution issues by adding explicit parameter definitions.
    """
    if "paths" not in openapi_schema:
        return openapi_schema

    for path, path_item in openapi_schema["paths"].items():
        # Extract path parameters from the URL
        path_params = _extract_path_parameters(path)

        if not path_params:
            continue

        # Add parameters to each operation in this path
        for method in ["get", "post", "put", "delete", "patch", "head", "options"]:
            if method in path_item and isinstance(path_item[method], dict):
                operation = path_item[method]
                if "parameters" not in operation:
                    operation["parameters"] = []

                # Add path parameters that aren't already defined
                existing_param_names = {p.get("name") for p in operation["parameters"] if p.get("in") == "path"}
                for param in path_params:
                    if param["name"] not in existing_param_names:
                        operation["parameters"].append(param)

    return openapi_schema


def _fix_schema_issues(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """Fix common schema issues: exclusiveMinimum and null defaults."""
    if "components" in openapi_schema and "schemas" in openapi_schema["components"]:
        for schema_def in openapi_schema["components"]["schemas"].values():
            _fix_schema_recursive(schema_def)
    return openapi_schema


def validate_openapi_schema(schema: dict[str, Any], schema_name: str = "OpenAPI schema") -> bool:
    """
    Validate an OpenAPI schema using openapi-spec-validator.

    Args:
        schema: The OpenAPI schema dictionary to validate
        schema_name: Name of the schema for error reporting

    Returns:
        True if valid, False otherwise

    Raises:
        OpenAPIValidationError: If validation fails
    """
    try:
        validate_spec(schema)
        print(f"✅ {schema_name} is valid")
        return True
    except OpenAPISpecValidatorError as e:
        print(f"❌ {schema_name} validation failed:")
        print(f"   {e}")
        return False
    except Exception as e:
        print(f"❌ {schema_name} validation error: {e}")
        return False


def _fix_schema_recursive(obj: Any) -> None:
    """Recursively fix schema issues: exclusiveMinimum and null defaults."""
    if isinstance(obj, dict):
        if "exclusiveMinimum" in obj and isinstance(obj["exclusiveMinimum"], int | float):
            obj["minimum"] = obj.pop("exclusiveMinimum")
        if "default" in obj and obj["default"] is None:
            del obj["default"]
            obj["nullable"] = True
        for value in obj.values():
            _fix_schema_recursive(value)
    elif isinstance(obj, list):
        for item in obj:
            _fix_schema_recursive(item)


def _clean_description(description: str) -> str:
    """Remove :param, :type, :returns, and other docstring metadata from description."""
    if not description:
        return description

    lines = description.split("\n")
    cleaned_lines = []
    skip_until_empty = False

    for line in lines:
        stripped = line.strip()
        # Skip lines that start with docstring metadata markers
        if stripped.startswith(
            (":param", ":type", ":return", ":returns", ":raises", ":exception", ":yield", ":yields", ":cvar")
        ):
            skip_until_empty = True
            continue
        # If we're skipping and hit an empty line, resume normal processing
        if skip_until_empty:
            if not stripped:
                skip_until_empty = False
            continue
        # Include the line if we're not skipping
        cleaned_lines.append(line)

    # Join and strip trailing whitespace
    result = "\n".join(cleaned_lines).strip()
    return result


def _clean_schema_descriptions(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """Clean descriptions in schema definitions by removing docstring metadata."""
    if "components" not in openapi_schema or "schemas" not in openapi_schema["components"]:
        return openapi_schema

    schemas = openapi_schema["components"]["schemas"]
    for schema_def in schemas.values():
        if isinstance(schema_def, dict) and "description" in schema_def and isinstance(schema_def["description"], str):
            schema_def["description"] = _clean_description(schema_def["description"])

    return openapi_schema


def _add_extra_body_params_extension(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Add x-llama-stack-extra-body-params extension to requestBody for endpoints with ExtraBodyField parameters.
    """
    if "paths" not in openapi_schema:
        return openapi_schema

    global _extra_body_fields

    from pydantic import TypeAdapter

    for path, path_item in openapi_schema["paths"].items():
        if not isinstance(path_item, dict):
            continue

        for method in ["get", "post", "put", "delete", "patch", "head", "options"]:
            if method not in path_item:
                continue

            operation = path_item[method]
            if not isinstance(operation, dict):
                continue

            # Check if we have extra body fields for this path/method
            key = (path, method.upper())
            if key not in _extra_body_fields:
                continue

            extra_body_params = _extra_body_fields[key]

            # Ensure requestBody exists
            if "requestBody" not in operation:
                continue

            request_body = operation["requestBody"]
            if not isinstance(request_body, dict):
                continue

            # Get the schema from requestBody
            content = request_body.get("content", {})
            json_content = content.get("application/json", {})
            schema_ref = json_content.get("schema", {})

            # Remove extra body fields from the schema if they exist as properties
            # Handle both $ref schemas and inline schemas
            if isinstance(schema_ref, dict):
                if "$ref" in schema_ref:
                    # Schema is a reference - remove from the referenced schema
                    ref_path = schema_ref["$ref"]
                    if ref_path.startswith("#/components/schemas/"):
                        schema_name = ref_path.split("/")[-1]
                        if "components" in openapi_schema and "schemas" in openapi_schema["components"]:
                            schema_def = openapi_schema["components"]["schemas"].get(schema_name)
                            if isinstance(schema_def, dict) and "properties" in schema_def:
                                for param_name, _, _ in extra_body_params:
                                    if param_name in schema_def["properties"]:
                                        del schema_def["properties"][param_name]
                                        # Also remove from required if present
                                        if "required" in schema_def and param_name in schema_def["required"]:
                                            schema_def["required"].remove(param_name)
                elif "properties" in schema_ref:
                    # Schema is inline - remove directly from it
                    for param_name, _, _ in extra_body_params:
                        if param_name in schema_ref["properties"]:
                            del schema_ref["properties"][param_name]
                            # Also remove from required if present
                            if "required" in schema_ref and param_name in schema_ref["required"]:
                                schema_ref["required"].remove(param_name)

            # Build the extra body params schema
            extra_params_schema = {}
            for param_name, param_type, description in extra_body_params:
                try:
                    # Generate JSON schema for the parameter type
                    adapter = TypeAdapter(param_type)
                    param_schema = adapter.json_schema(ref_template="#/components/schemas/{model}")

                    # Add description if provided
                    if description:
                        param_schema["description"] = description

                    extra_params_schema[param_name] = param_schema
                except Exception:
                    # If we can't generate schema, skip this parameter
                    continue

            if extra_params_schema:
                # Add the extension to requestBody
                if "x-llama-stack-extra-body-params" not in request_body:
                    request_body["x-llama-stack-extra-body-params"] = extra_params_schema

    return openapi_schema


def _remove_query_params_from_body_endpoints(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Remove query parameters from POST/PUT/PATCH endpoints that have a request body.
    FastAPI sometimes infers parameters as query params even when they should be in the request body.
    """
    if "paths" not in openapi_schema:
        return openapi_schema

    body_methods = {"post", "put", "patch"}

    for _path, path_item in openapi_schema["paths"].items():
        if not isinstance(path_item, dict):
            continue

        for method in body_methods:
            if method not in path_item:
                continue

            operation = path_item[method]
            if not isinstance(operation, dict):
                continue

            # Check if this operation has a request body
            has_request_body = "requestBody" in operation and operation["requestBody"]

            if has_request_body:
                # Remove all query parameters (parameters with "in": "query")
                if "parameters" in operation:
                    # Filter out query parameters, keep path and header parameters
                    operation["parameters"] = [
                        param
                        for param in operation["parameters"]
                        if isinstance(param, dict) and param.get("in") != "query"
                    ]
                    # Remove the parameters key if it's now empty
                    if not operation["parameters"]:
                        del operation["parameters"]

    return openapi_schema


def _remove_request_bodies_from_get_endpoints(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Remove request bodies from GET endpoints and convert their parameters to query parameters.

    GET requests should never have request bodies - all parameters should be query parameters.
    This function removes any requestBody that FastAPI may have incorrectly added to GET endpoints
    and converts any parameters in the requestBody to query parameters.
    """
    if "paths" not in openapi_schema:
        return openapi_schema

    for _path, path_item in openapi_schema["paths"].items():
        if not isinstance(path_item, dict):
            continue

        # Check GET method specifically
        if "get" in path_item:
            operation = path_item["get"]
            if not isinstance(operation, dict):
                continue

            if "requestBody" in operation:
                request_body = operation["requestBody"]
                # Extract parameters from requestBody and convert to query parameters
                if isinstance(request_body, dict) and "content" in request_body:
                    content = request_body.get("content", {})
                    json_content = content.get("application/json", {})
                    schema = json_content.get("schema", {})

                    if "parameters" not in operation:
                        operation["parameters"] = []
                    elif not isinstance(operation["parameters"], list):
                        operation["parameters"] = []

                    # If the schema has properties, convert each to a query parameter
                    if isinstance(schema, dict) and "properties" in schema:
                        for param_name, param_schema in schema["properties"].items():
                            # Check if this parameter is already in the parameters list
                            existing_param = None
                            for existing in operation["parameters"]:
                                if isinstance(existing, dict) and existing.get("name") == param_name:
                                    existing_param = existing
                                    break

                            if not existing_param:
                                # Create a new query parameter from the requestBody property
                                required = param_name in schema.get("required", [])
                                query_param = {
                                    "name": param_name,
                                    "in": "query",
                                    "required": required,
                                    "schema": param_schema,
                                }
                                # Add description if present
                                if "description" in param_schema:
                                    query_param["description"] = param_schema["description"]
                                operation["parameters"].append(query_param)
                    elif isinstance(schema, dict):
                        # Handle direct schema (not a model with properties)
                        # Try to infer parameter name from schema title
                        param_name = schema.get("title", "").lower().replace(" ", "_")
                        if param_name:
                            # Check if this parameter is already in the parameters list
                            existing_param = None
                            for existing in operation["parameters"]:
                                if isinstance(existing, dict) and existing.get("name") == param_name:
                                    existing_param = existing
                                    break

                            if not existing_param:
                                # Create a new query parameter from the requestBody schema
                                query_param = {
                                    "name": param_name,
                                    "in": "query",
                                    "required": False,  # Default to optional for GET requests
                                    "schema": schema,
                                }
                                # Add description if present
                                if "description" in schema:
                                    query_param["description"] = schema["description"]
                                operation["parameters"].append(query_param)

                # Remove request body from GET endpoint
                del operation["requestBody"]

    return openapi_schema


def _convert_multiline_strings_to_literal(obj: Any) -> Any:
    """Recursively convert multi-line strings to LiteralScalarString for YAML block scalar formatting."""
    try:
        from ruamel.yaml.scalarstring import LiteralScalarString

        if isinstance(obj, str) and "\n" in obj:
            return LiteralScalarString(obj)
        elif isinstance(obj, dict):
            return {key: _convert_multiline_strings_to_literal(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [_convert_multiline_strings_to_literal(item) for item in obj]
        else:
            return obj
    except ImportError:
        return obj


def _write_yaml_file(file_path: Path, schema: dict[str, Any]) -> None:
    """Write schema to YAML file using ruamel.yaml if available, otherwise standard yaml."""
    try:
        from ruamel.yaml import YAML

        yaml_writer = YAML()
        yaml_writer.default_flow_style = False
        yaml_writer.sort_keys = False
        yaml_writer.width = 4096
        yaml_writer.allow_unicode = True
        schema = _convert_multiline_strings_to_literal(schema)
        with open(file_path, "w") as f:
            yaml_writer.dump(schema, f)
    except ImportError:
        with open(file_path, "w") as f:
            yaml.dump(schema, f, default_flow_style=False, sort_keys=False)


def _get_explicit_schema_names(openapi_schema: dict[str, Any]) -> set[str]:
    """Get all registered schema names and @json_schema_type decorated model names."""
    from llama_stack.schema_utils import _registered_schemas

    registered_schema_names = {info["name"] for info in _registered_schemas.values()}
    json_schema_type_names = _get_all_json_schema_type_names()
    return registered_schema_names | json_schema_type_names


def _add_transitive_references(
    referenced_schemas: set[str], all_schemas: dict[str, Any], initial_schemas: set[str] | None = None
) -> set[str]:
    """Add transitive references for given schemas."""
    if initial_schemas:
        referenced_schemas.update(initial_schemas)
        additional_schemas = set()
        for schema_name in initial_schemas:
            if schema_name in all_schemas:
                additional_schemas.update(_find_schema_refs_in_object(all_schemas[schema_name]))
    else:
        additional_schemas = set()
        for schema_name in referenced_schemas:
            if schema_name in all_schemas:
                additional_schemas.update(_find_schema_refs_in_object(all_schemas[schema_name]))

    while additional_schemas:
        new_schemas = additional_schemas - referenced_schemas
        if not new_schemas:
            break
        referenced_schemas.update(new_schemas)
        additional_schemas = set()
        for schema_name in new_schemas:
            if schema_name in all_schemas:
                additional_schemas.update(_find_schema_refs_in_object(all_schemas[schema_name]))

    return referenced_schemas


def _filter_schemas_by_references(
    filtered_schema: dict[str, Any], filtered_paths: dict[str, Any], openapi_schema: dict[str, Any]
) -> dict[str, Any]:
    """Filter schemas to only include ones referenced by filtered paths and explicit schemas."""
    if "components" not in filtered_schema or "schemas" not in filtered_schema["components"]:
        return filtered_schema

    referenced_schemas = _find_schemas_referenced_by_paths(filtered_paths, openapi_schema)
    all_schemas = openapi_schema.get("components", {}).get("schemas", {})
    explicit_schema_names = _get_explicit_schema_names(openapi_schema)
    referenced_schemas = _add_transitive_references(referenced_schemas, all_schemas, explicit_schema_names)

    filtered_schemas = {
        name: schema for name, schema in filtered_schema["components"]["schemas"].items() if name in referenced_schemas
    }
    filtered_schema["components"]["schemas"] = filtered_schemas

    if "components" in openapi_schema and "$defs" in openapi_schema["components"]:
        if "components" not in filtered_schema:
            filtered_schema["components"] = {}
        filtered_schema["components"]["$defs"] = openapi_schema["components"]["$defs"]

    return filtered_schema


def _filter_schema_by_version(
    openapi_schema: dict[str, Any], stable_only: bool = True, exclude_deprecated: bool = True
) -> dict[str, Any]:
    """
    Filter OpenAPI schema by API version.

    Args:
        openapi_schema: The full OpenAPI schema
        stable_only: If True, return only /v1/ paths (stable). If False, return only /v1alpha/ and /v1beta/ paths (experimental).
        exclude_deprecated: If True, exclude deprecated endpoints from the result.

    Returns:
        Filtered OpenAPI schema
    """
    filtered_schema = openapi_schema.copy()

    if "paths" not in filtered_schema:
        return filtered_schema

    filtered_paths = {}
    for path, path_item in filtered_schema["paths"].items():
        if not isinstance(path_item, dict):
            continue

        # Filter at operation level, not path level
        # This allows paths with both deprecated and non-deprecated operations
        filtered_path_item = {}
        for method in ["get", "post", "put", "delete", "patch", "head", "options"]:
            if method not in path_item:
                continue
            operation = path_item[method]
            if not isinstance(operation, dict):
                continue

            # Skip deprecated operations if exclude_deprecated is True
            if exclude_deprecated and operation.get("deprecated", False):
                continue

            filtered_path_item[method] = operation

        # Only include path if it has at least one operation after filtering
        if filtered_path_item:
            # Check if path matches version filter
            if (stable_only and _is_stable_path(path)) or (not stable_only and _is_experimental_path(path)):
                filtered_paths[path] = filtered_path_item

    filtered_schema["paths"] = filtered_paths
    return _filter_schemas_by_references(filtered_schema, filtered_paths, openapi_schema)


def _find_schemas_referenced_by_paths(filtered_paths: dict[str, Any], openapi_schema: dict[str, Any]) -> set[str]:
    """
    Find all schemas that are referenced by the filtered paths.
    This recursively traverses the path definitions to find all $ref references.
    """
    referenced_schemas = set()

    # Traverse all filtered paths
    for _, path_item in filtered_paths.items():
        if not isinstance(path_item, dict):
            continue

        # Check each HTTP method in the path
        for method in ["get", "post", "put", "delete", "patch", "head", "options"]:
            if method in path_item:
                operation = path_item[method]
                if isinstance(operation, dict):
                    # Find all schema references in this operation
                    referenced_schemas.update(_find_schema_refs_in_object(operation))

    # Also check the responses section for schema references
    if "components" in openapi_schema and "responses" in openapi_schema["components"]:
        referenced_schemas.update(_find_schema_refs_in_object(openapi_schema["components"]["responses"]))

    # Also include schemas that are referenced by other schemas (transitive references)
    # This ensures we include all dependencies
    all_schemas = openapi_schema.get("components", {}).get("schemas", {})
    additional_schemas = set()

    for schema_name in referenced_schemas:
        if schema_name in all_schemas:
            additional_schemas.update(_find_schema_refs_in_object(all_schemas[schema_name]))

    # Keep adding transitive references until no new ones are found
    while additional_schemas:
        new_schemas = additional_schemas - referenced_schemas
        if not new_schemas:
            break
        referenced_schemas.update(new_schemas)
        additional_schemas = set()
        for schema_name in new_schemas:
            if schema_name in all_schemas:
                additional_schemas.update(_find_schema_refs_in_object(all_schemas[schema_name]))

    return referenced_schemas


def _find_schema_refs_in_object(obj: Any) -> set[str]:
    """
    Recursively find all schema references ($ref) in an object.
    """
    refs = set()

    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "$ref" and isinstance(value, str) and value.startswith("#/components/schemas/"):
                schema_name = value.split("/")[-1]
                refs.add(schema_name)
            else:
                refs.update(_find_schema_refs_in_object(value))
    elif isinstance(obj, list):
        for item in obj:
            refs.update(_find_schema_refs_in_object(item))

    return refs


def _get_all_json_schema_type_names() -> set[str]:
    """
    Get all schema names from @json_schema_type decorated models.
    This ensures they are included in filtered schemas even if not directly referenced by paths.
    """
    schema_names = set()
    apis_modules = _import_all_modules_in_package("llama_stack.apis")
    for module in apis_modules:
        for attr_name in dir(module):
            try:
                attr = getattr(module, attr_name)
                if (
                    hasattr(attr, "_llama_stack_schema_type")
                    and hasattr(attr, "model_json_schema")
                    and hasattr(attr, "__name__")
                ):
                    schema_names.add(attr.__name__)
            except (AttributeError, TypeError):
                continue
    return schema_names


def _is_path_deprecated(path_item: dict[str, Any]) -> bool:
    """Check if a path item has any deprecated operations."""
    if not isinstance(path_item, dict):
        return False
    for method in ["get", "post", "put", "delete", "patch", "head", "options"]:
        if isinstance(path_item.get(method), dict) and path_item[method].get("deprecated", False):
            return True
    return False


def _path_starts_with_version(path: str, version: str) -> bool:
    """Check if a path starts with a specific API version prefix."""
    return path.startswith(f"/{version}/")


def _is_stable_path(path: str) -> bool:
    """Check if a path is a stable v1 path (not v1alpha or v1beta)."""
    return (
        _path_starts_with_version(path, LLAMA_STACK_API_V1)
        and not _path_starts_with_version(path, LLAMA_STACK_API_V1ALPHA)
        and not _path_starts_with_version(path, LLAMA_STACK_API_V1BETA)
    )


def _is_experimental_path(path: str) -> bool:
    """Check if a path is an experimental path (v1alpha or v1beta)."""
    return _path_starts_with_version(path, LLAMA_STACK_API_V1ALPHA) or _path_starts_with_version(
        path, LLAMA_STACK_API_V1BETA
    )


def _filter_deprecated_schema(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Filter OpenAPI schema to include only deprecated endpoints.
    Includes all deprecated endpoints regardless of version (v1, v1alpha, v1beta).
    """
    filtered_schema = openapi_schema.copy()

    if "paths" not in filtered_schema:
        return filtered_schema

    # Filter paths to only include deprecated ones
    filtered_paths = {}
    for path, path_item in filtered_schema["paths"].items():
        if _is_path_deprecated(path_item):
            filtered_paths[path] = path_item

    filtered_schema["paths"] = filtered_paths

    return filtered_schema


def _filter_combined_schema(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Filter OpenAPI schema to include both stable (v1) and experimental (v1alpha, v1beta) APIs.
    Includes deprecated endpoints. This is used for the combined "stainless" spec.
    """
    filtered_schema = openapi_schema.copy()

    if "paths" not in filtered_schema:
        return filtered_schema

    # Filter paths to include stable (v1) and experimental (v1alpha, v1beta), including deprecated
    filtered_paths = {}
    for path, path_item in filtered_schema["paths"].items():
        if not isinstance(path_item, dict):
            continue

        # Include all operations (both deprecated and non-deprecated) for the combined spec
        # Filter at operation level to preserve the structure
        filtered_path_item = {}
        for method in ["get", "post", "put", "delete", "patch", "head", "options"]:
            if method not in path_item:
                continue
            operation = path_item[method]
            if not isinstance(operation, dict):
                continue

            # Include all operations, including deprecated ones
            filtered_path_item[method] = operation

        # Only include path if it has at least one operation
        if filtered_path_item:
            # Check if path matches version filter (stable or experimental)
            if _is_stable_path(path) or _is_experimental_path(path):
                filtered_paths[path] = filtered_path_item

    filtered_schema["paths"] = filtered_paths

    return _filter_schemas_by_references(filtered_schema, filtered_paths, openapi_schema)


def generate_openapi_spec(output_dir: str) -> dict[str, Any]:
    """
    Generate OpenAPI specification using FastAPI's built-in method.

    Args:
        output_dir: Directory to save the generated files

    Returns:
        The generated OpenAPI specification as a dictionary
    """
    # Create the FastAPI app
    app = create_llama_stack_app()

    # Generate the OpenAPI schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        servers=app.servers,
    )

    # Set OpenAPI version to 3.1.0
    openapi_schema["openapi"] = "3.1.0"

    # Add standard error responses
    openapi_schema = _add_error_responses(openapi_schema)

    # Ensure all @json_schema_type decorated models are included
    openapi_schema = _ensure_json_schema_types_included(openapi_schema)

    # Fix $ref references to point to components/schemas instead of $defs
    openapi_schema = _fix_ref_references(openapi_schema)

    # Fix path parameter resolution issues
    openapi_schema = _fix_path_parameters(openapi_schema)

    # Eliminate $defs section entirely for oasdiff compatibility
    openapi_schema = _eliminate_defs_section(openapi_schema)

    # Clean descriptions in schema definitions by removing docstring metadata
    openapi_schema = _clean_schema_descriptions(openapi_schema)

    # Remove query parameters from POST/PUT/PATCH endpoints that have a request body
    # FastAPI sometimes infers parameters as query params even when they should be in the request body
    openapi_schema = _remove_query_params_from_body_endpoints(openapi_schema)

    # Add x-llama-stack-extra-body-params extension for ExtraBodyField parameters
    openapi_schema = _add_extra_body_params_extension(openapi_schema)

    # Remove request bodies from GET endpoints (GET requests should never have request bodies)
    # This must run AFTER _add_extra_body_params_extension to ensure any request bodies
    # that FastAPI incorrectly added to GET endpoints are removed
    openapi_schema = _remove_request_bodies_from_get_endpoints(openapi_schema)

    # Split into stable (v1 only), experimental (v1alpha + v1beta), deprecated, and combined (stainless) specs
    # Each spec needs its own deep copy of the full schema to avoid cross-contamination
    import copy

    stable_schema = _filter_schema_by_version(copy.deepcopy(openapi_schema), stable_only=True, exclude_deprecated=True)
    experimental_schema = _filter_schema_by_version(
        copy.deepcopy(openapi_schema), stable_only=False, exclude_deprecated=True
    )
    deprecated_schema = _filter_deprecated_schema(copy.deepcopy(openapi_schema))
    combined_schema = _filter_combined_schema(copy.deepcopy(openapi_schema))

    base_description = (
        "This is the specification of the Llama Stack that provides\n"
        "                    a set of endpoints and their corresponding interfaces that are\n"
        "    tailored to\n"
        "                    best leverage Llama Models."
    )

    schema_configs = [
        (
            stable_schema,
            "Llama Stack Specification",
            "**✅ STABLE**: Production-ready APIs with backward compatibility guarantees.",
        ),
        (
            experimental_schema,
            "Llama Stack Specification - Experimental APIs",
            "**🧪 EXPERIMENTAL**: Pre-release APIs (v1alpha, v1beta) that may change before\n    becoming stable.",
        ),
        (
            deprecated_schema,
            "Llama Stack Specification - Deprecated APIs",
            "**⚠️ DEPRECATED**: Legacy APIs that may be removed in future versions. Use for\n    migration reference only.",
        ),
        (
            combined_schema,
            "Llama Stack Specification - Stable & Experimental APIs",
            "**🔗 COMBINED**: This specification includes both stable production-ready APIs\n    and experimental pre-release APIs. Use stable APIs for production deployments\n    and experimental APIs for testing new features.",
        ),
    ]

    for schema, title, description_suffix in schema_configs:
        if "info" not in schema:
            schema["info"] = {}
        schema["info"].update(
            {
                "title": title,
                "version": "v1",
                "description": f"{base_description}\n\n    {description_suffix}",
            }
        )

    schemas_to_validate = [
        (stable_schema, "Stable schema"),
        (experimental_schema, "Experimental schema"),
        (deprecated_schema, "Deprecated schema"),
        (combined_schema, "Combined (stainless) schema"),
    ]

    for schema, _ in schemas_to_validate:
        _fix_schema_issues(schema)

    print("\n🔍 Validating generated schemas...")
    failed_schemas = [name for schema, name in schemas_to_validate if not validate_openapi_schema(schema, name)]
    if failed_schemas:
        raise ValueError(f"Invalid schemas: {', '.join(failed_schemas)}")

    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save the stable specification
    yaml_path = output_path / "llama-stack-spec.yaml"
    _write_yaml_file(yaml_path, stable_schema)
    # Post-process the YAML file to remove $defs section and fix references
    with open(yaml_path) as f:
        yaml_content = f.read()

    if "  $defs:" in yaml_content or "#/$defs/" in yaml_content:
        # Use string replacement to fix references directly
        if "#/$defs/" in yaml_content:
            yaml_content = yaml_content.replace("#/$defs/", "#/components/schemas/")

        # Parse the YAML content
        yaml_data = yaml.safe_load(yaml_content)

        # Move $defs to components/schemas if it exists
        if "$defs" in yaml_data:
            if "components" not in yaml_data:
                yaml_data["components"] = {}
            if "schemas" not in yaml_data["components"]:
                yaml_data["components"]["schemas"] = {}

            # Move all $defs to components/schemas
            for def_name, def_schema in yaml_data["$defs"].items():
                yaml_data["components"]["schemas"][def_name] = def_schema

            # Remove the $defs section
            del yaml_data["$defs"]

        # Write the modified YAML back
        _write_yaml_file(yaml_path, yaml_data)

    print(f"✅ Generated YAML (stable): {yaml_path}")

    experimental_yaml_path = output_path / "experimental-llama-stack-spec.yaml"
    _write_yaml_file(experimental_yaml_path, experimental_schema)
    print(f"✅ Generated YAML (experimental): {experimental_yaml_path}")

    deprecated_yaml_path = output_path / "deprecated-llama-stack-spec.yaml"
    _write_yaml_file(deprecated_yaml_path, deprecated_schema)
    print(f"✅ Generated YAML (deprecated): {deprecated_yaml_path}")

    # Generate combined (stainless) spec
    stainless_yaml_path = output_path / "stainless-llama-stack-spec.yaml"
    _write_yaml_file(stainless_yaml_path, combined_schema)
    print(f"✅ Generated YAML (stainless/combined): {stainless_yaml_path}")

    return stable_schema


def main():
    """Main entry point for the FastAPI OpenAPI generator."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate OpenAPI specification using FastAPI")
    parser.add_argument("output_dir", help="Output directory for generated files")

    args = parser.parse_args()

    print("🚀 Generating OpenAPI specification using FastAPI...")
    print(f"📁 Output directory: {args.output_dir}")

    try:
        openapi_schema = generate_openapi_spec(output_dir=args.output_dir)

        print("\n✅ OpenAPI specification generated successfully!")
        print(f"📊 Schemas: {len(openapi_schema.get('components', {}).get('schemas', {}))}")
        print(f"🛣️  Paths: {len(openapi_schema.get('paths', {}))}")
        operation_count = sum(
            1
            for path_info in openapi_schema.get("paths", {}).values()
            for method in ["get", "post", "put", "delete", "patch"]
            if method in path_info
        )
        print(f"🔧 Operations: {operation_count}")

    except Exception as e:
        print(f"❌ Error generating OpenAPI specification: {e}")
        raise


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
FastAPI-based OpenAPI generator for Llama Stack.
"""

import inspect
import json
from pathlib import Path
from typing import Annotated, Any, Literal, get_args, get_origin

import yaml
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from llama_stack.apis.datatypes import Api
from llama_stack.core.resolver import api_protocol_map

# Import the existing route discovery system
from llama_stack.core.server.routes import get_all_api_routes


def _get_all_api_routes_with_functions():
    """
    Get all API routes with their actual function references.
    This is a modified version of get_all_api_routes that includes the function.
    """
    from aiohttp import hdrs
    from starlette.routing import Route

    from llama_stack.apis.tools import RAGToolRuntime, SpecialToolGroup

    apis = {}
    protocols = api_protocol_map()
    toolgroup_protocols = {
        SpecialToolGroup.rag_tool: RAGToolRuntime,
    }

    for api, protocol in protocols.items():
        routes = []
        protocol_methods = inspect.getmembers(protocol, predicate=inspect.isfunction)

        # HACK ALERT
        if api == Api.tool_runtime:
            for tool_group in SpecialToolGroup:
                sub_protocol = toolgroup_protocols[tool_group]
                sub_protocol_methods = inspect.getmembers(sub_protocol, predicate=inspect.isfunction)
                for name, method in sub_protocol_methods:
                    if not hasattr(method, "__webmethod__"):
                        continue
                    protocol_methods.append((f"{tool_group.value}.{name}", method))

        for name, method in protocol_methods:
            # Get all webmethods for this method (supports multiple decorators)
            webmethods = getattr(method, "__webmethods__", [])
            if not webmethods:
                continue

            # Create routes for each webmethod decorator
            for webmethod in webmethods:
                path = f"/{webmethod.level}/{webmethod.route.lstrip('/')}"
                if webmethod.method == hdrs.METH_GET:
                    http_method = hdrs.METH_GET
                elif webmethod.method == hdrs.METH_DELETE:
                    http_method = hdrs.METH_DELETE
                else:
                    http_method = hdrs.METH_POST

                # Store the function reference in the webmethod
                webmethod.func = method

                routes.append((Route(path=path, methods=[http_method], name=name, endpoint=None), webmethod))

        apis[api] = routes

    return apis


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
            {"url": "https://api.llamastack.com", "description": "Production server"},
            {"url": "https://staging-api.llamastack.com", "description": "Staging server"},
        ],
    )

    # Get all API routes using the modified system that includes functions
    api_routes = _get_all_api_routes_with_functions()

    # Create FastAPI routes from the discovered routes
    for _, routes in api_routes.items():
        for route, webmethod in routes:
            # Convert the route to a FastAPI endpoint
            _create_fastapi_endpoint(app, route, webmethod)

    return app


def _create_fastapi_endpoint(app: FastAPI, route, webmethod):
    """
    Create a FastAPI endpoint from a discovered route and webmethod.
    This creates endpoints with actual Pydantic models for proper schema generation.
    """
    # Extract route information
    path = route.path
    methods = route.methods
    name = route.name

    # Convert path parameters from {param} to {param:path} format for FastAPI
    fastapi_path = path.replace("{", "{").replace("}", "}")

    # Try to find actual models for this endpoint
    request_model, response_model, query_parameters = _find_models_for_endpoint(webmethod)

    # Extract response description from webmethod docstring (always try this first)
    response_description = _extract_response_description_from_docstring(webmethod, response_model)

    # Create endpoint function with proper typing
    if request_model and response_model:
        # POST/PUT request with request body
        async def typed_endpoint(request: request_model) -> response_model:
            """Typed endpoint for proper schema generation."""
            return response_model()

        endpoint_func = typed_endpoint
    elif response_model and query_parameters:
        # Request with individual parameters (could be GET with query params or POST with individual params)
        # Create a function with the actual query parameters
        def create_query_endpoint_func():
            # Build the function signature dynamically
            import inspect

            # Create parameter annotations
            param_annotations = {}
            param_defaults = {}

            for param_name, param_type, default_value in query_parameters:
                # Handle problematic type annotations that cause FastAPI issues
                safe_type = _make_type_safe_for_fastapi(param_type)
                param_annotations[param_name] = safe_type
                if default_value is not None:
                    param_defaults[param_name] = default_value

            # Create the function signature
            sig = inspect.Signature(
                [
                    inspect.Parameter(
                        name=param_name,
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=default_value,
                        annotation=param_annotations[param_name],
                    )
                    for param_name, param_type, default_value in query_parameters
                ]
            )

            async def query_endpoint(**kwargs) -> response_model:
                """Query endpoint for proper schema generation."""
                return response_model()

            # Set the signature
            query_endpoint.__signature__ = sig
            query_endpoint.__annotations__ = param_annotations
            return query_endpoint

        endpoint_func = create_query_endpoint_func()
    elif response_model:
        # Response-only endpoint (no parameters)
        async def response_only_endpoint() -> response_model:
            """Response-only endpoint for proper schema generation."""
            return response_model()

        endpoint_func = response_only_endpoint
    else:
        # Fallback to generic endpoint
        async def generic_endpoint(*args, **kwargs):
            """Generic endpoint - this would be replaced with actual implementation."""
            return {"message": f"Endpoint {name} not implemented in OpenAPI generator"}

        endpoint_func = generic_endpoint

    # Add the endpoint to the FastAPI app
    is_deprecated = webmethod.deprecated or False
    route_kwargs = {
        "name": name,
        "tags": [_get_tag_from_api(webmethod)],
        "deprecated": is_deprecated,
        "responses": {
            200: {
                "description": response_description,
                "content": {
                    "application/json": {
                        "schema": {"$ref": f"#/components/schemas/{response_model.__name__}"} if response_model else {}
                    }
                },
            },
            400: {"$ref": "#/components/responses/BadRequest400"},
            429: {"$ref": "#/components/responses/TooManyRequests429"},
            500: {"$ref": "#/components/responses/InternalServerError500"},
            "default": {"$ref": "#/components/responses/DefaultError"},
        },
    }

    for method in methods:
        if method.upper() == "GET":
            app.get(fastapi_path, **route_kwargs)(endpoint_func)
        elif method.upper() == "POST":
            app.post(fastapi_path, **route_kwargs)(endpoint_func)
        elif method.upper() == "PUT":
            app.put(fastapi_path, **route_kwargs)(endpoint_func)
        elif method.upper() == "DELETE":
            app.delete(fastapi_path, **route_kwargs)(endpoint_func)
        elif method.upper() == "PATCH":
            app.patch(fastapi_path, **route_kwargs)(endpoint_func)


def _extract_response_description_from_docstring(webmethod, response_model) -> str:
    """
    Extract response description from the actual function docstring.
    Looks for :returns: in the docstring and uses that as the description.
    """
    # Try to get the actual function from the webmethod
    # The webmethod should have a reference to the original function
    func = getattr(webmethod, "func", None)
    if not func:
        # If we can't get the function, return a generic description
        return "Successful Response"

    # Get the function's docstring
    docstring = func.__doc__ or ""

    # Look for :returns: line in the docstring
    lines = docstring.split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith(":returns:"):
            # Extract the description after :returns:
            description = line[9:].strip()  # Remove ':returns:' prefix
            if description:
                return description

    # If no :returns: found, return a generic description
    return "Successful Response"


def _get_tag_from_api(webmethod) -> str:
    """Extract a tag name from the webmethod for API grouping."""
    # Extract API name from the route path
    if webmethod.level:
        return webmethod.level.replace("/", "").title()
    return "API"


def _find_models_for_endpoint(webmethod) -> tuple[type | None, type | None, list[tuple[str, type, Any]]]:
    """
    Find appropriate request and response models for an endpoint by analyzing the actual function signature.
    This uses the webmethod's function to determine the correct models dynamically.

    Returns:
        tuple: (request_model, response_model, query_parameters)
        where query_parameters is a list of (name, type, default_value) tuples
    """
    try:
        # Get the actual function from the webmethod
        func = getattr(webmethod, "func", None)
        if not func:
            return None, None, []

        # Analyze the function signature
        sig = inspect.signature(func)

        # Find request model (first parameter that's not 'self')
        request_model = None
        query_parameters = []

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            # Check if it's a Pydantic model (for POST/PUT requests)
            param_type = param.annotation
            if hasattr(param_type, "model_json_schema"):
                request_model = param_type
                break
            elif get_origin(param_type) is Annotated:
                # Handle Annotated types - get the base type
                args = get_args(param_type)
                if args and hasattr(args[0], "model_json_schema"):
                    request_model = args[0]
                    break
            else:
                # This is likely a query parameter for GET requests
                # Store the parameter info for later use
                default_value = param.default if param.default != inspect.Parameter.empty else None

                # Extract the base type from union types (e.g., str | None -> str)
                # Also make it safe for FastAPI to avoid forward reference issues
                base_type = _make_type_safe_for_fastapi(param_type)
                query_parameters.append((param_name, base_type, default_value))

        # Find response model from return annotation
        response_model = None
        return_annotation = sig.return_annotation
        if return_annotation != inspect.Signature.empty:
            if hasattr(return_annotation, "model_json_schema"):
                response_model = return_annotation
            elif get_origin(return_annotation) is Annotated:
                # Handle Annotated return types
                args = get_args(return_annotation)
                if args and hasattr(args[0], "model_json_schema"):
                    response_model = args[0]
            elif get_origin(return_annotation) is type(return_annotation):  # Union type
                # Handle union types - try to find the first Pydantic model
                args = get_args(return_annotation)
                for arg in args:
                    if hasattr(arg, "model_json_schema"):
                        response_model = arg
                        break

        return request_model, response_model, query_parameters

    except Exception:
        # If we can't analyze the function signature, return None
        return None, None, []


def _make_type_safe_for_fastapi(type_hint) -> type:
    """
    Make a type hint safe for FastAPI by converting problematic types to their base types.
    This handles cases like Literal["24h"] that cause forward reference errors.
    """
    # Handle Literal types that might cause issues
    if hasattr(type_hint, "__origin__") and type_hint.__origin__ is Literal:
        args = get_args(type_hint)
        if args:
            # Get the type of the first literal value
            first_arg = args[0]
            if isinstance(first_arg, str):
                return str
            elif isinstance(first_arg, int):
                return int
            elif isinstance(first_arg, float):
                return float
            elif isinstance(first_arg, bool):
                return bool
            else:
                return type(first_arg)

    # Handle Union types (Python 3.10+ uses | syntax)
    origin = get_origin(type_hint)

    if origin is type(None) or (origin is type and type_hint is type(None)):
        # This is just None, return None
        return type_hint

    # Handle Union types (both old Union and new | syntax)
    if origin is type(type_hint) or (hasattr(type_hint, "__args__") and type_hint.__args__):
        # This is a union type, find the non-None type
        args = get_args(type_hint)
        for arg in args:
            if arg is not type(None) and arg is not None:
                return arg
        # If all args are None, return the first one
        return args[0] if args else type_hint

    # Not a union type, return as-is
    return type_hint


def _generate_schema_for_type(type_hint) -> dict[str, Any]:
    """
    Generate a JSON schema for a given type hint.
    This is a simplified version that handles basic types.
    """
    # Handle Union types (e.g., str | None)
    if get_origin(type_hint) is type(None) or (get_origin(type_hint) is type and type_hint is type(None)):
        return {"type": "null"}

    # Handle list types
    if get_origin(type_hint) is list:
        args = get_args(type_hint)
        if args:
            item_type = args[0]
            return {"type": "array", "items": _generate_schema_for_type(item_type)}
        return {"type": "array"}

    # Handle basic types
    if type_hint is str:
        return {"type": "string"}
    elif type_hint is int:
        return {"type": "integer"}
    elif type_hint is float:
        return {"type": "number"}
    elif type_hint is bool:
        return {"type": "boolean"}
    elif type_hint is dict:
        return {"type": "object"}
    elif type_hint is list:
        return {"type": "array"}

    # For complex types, try to get the schema from Pydantic
    try:
        if hasattr(type_hint, "model_json_schema"):
            return type_hint.model_json_schema()
        elif hasattr(type_hint, "__name__"):
            return {"$ref": f"#/components/schemas/{type_hint.__name__}"}
    except Exception:
        pass

    # Fallback
    return {"type": "object"}


def _add_llama_stack_extensions(openapi_schema: dict[str, Any], app: FastAPI) -> dict[str, Any]:
    """
    Add Llama Stack specific extensions to the OpenAPI schema.
    This includes x-llama-stack-extra-body-params for ExtraBodyField parameters.
    """
    # Get all API routes to find functions with ExtraBodyField parameters
    api_routes = get_all_api_routes()

    for api_name, routes in api_routes.items():
        for route, webmethod in routes:
            # Extract path and method
            path = route.path
            methods = route.methods

            for method in methods:
                method_lower = method.lower()
                if method_lower in openapi_schema.get("paths", {}).get(path, {}):
                    operation = openapi_schema["paths"][path][method_lower]

                    # Try to find the actual function that implements this route
                    # and extract its ExtraBodyField parameters
                    extra_body_params = _find_extra_body_params_for_route(api_name, route, webmethod)

                    if extra_body_params:
                        operation["x-llama-stack-extra-body-params"] = extra_body_params

    return openapi_schema


def _find_extra_body_params_for_route(api_name: str, route, webmethod) -> list[dict[str, Any]]:
    """
    Find the actual function that implements a route and extract its ExtraBodyField parameters.
    """
    try:
        # Try to get the actual function from the API protocol map
        from llama_stack.core.resolver import api_protocol_map

        # Look up the API implementation
        if api_name in api_protocol_map:
            _ = api_protocol_map[api_name]

            # Try to find the method that matches this route
            # This is a simplified approach - we'd need to map the route to the actual method
            # For now, we'll return an empty list to avoid hardcoding
            return []

        return []
    except Exception:
        # If we can't find the function, return empty list
        return []


def _add_error_responses(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Add standard error response definitions to the OpenAPI schema.
    Uses the actual Error model from the codebase for consistency.
    """
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}

    if "responses" not in openapi_schema["components"]:
        openapi_schema["components"]["responses"] = {}

    # Import the actual Error model
    try:
        from llama_stack.apis.datatypes import Error

        # Generate the Error schema using Pydantic
        error_schema = Error.model_json_schema()

        # Ensure the Error schema is in the components/schemas
        if "schemas" not in openapi_schema["components"]:
            openapi_schema["components"]["schemas"] = {}

        # Only add Error schema if it doesn't already exist
        if "Error" not in openapi_schema["components"]["schemas"]:
            openapi_schema["components"]["schemas"]["Error"] = error_schema

    except ImportError:
        # Fallback if we can't import the Error model
        error_schema = {"$ref": "#/components/schemas/Error"}

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


def _fix_schema_issues(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Fix common schema issues that cause OpenAPI validation problems.
    This includes converting exclusiveMinimum numbers to minimum values.
    """
    if "components" not in openapi_schema or "schemas" not in openapi_schema["components"]:
        return openapi_schema

    schemas = openapi_schema["components"]["schemas"]

    # Fix exclusiveMinimum issues
    for _, schema_def in schemas.items():
        _fix_exclusive_minimum_in_schema(schema_def)

    return openapi_schema


def _fix_exclusive_minimum_in_schema(obj: Any) -> None:
    """
    Recursively fix exclusiveMinimum issues in a schema object.
    Converts exclusiveMinimum numbers to minimum values.
    """
    if isinstance(obj, dict):
        # Check if this is a schema with exclusiveMinimum
        if "exclusiveMinimum" in obj and isinstance(obj["exclusiveMinimum"], int | float):
            # Convert exclusiveMinimum number to minimum
            obj["minimum"] = obj["exclusiveMinimum"]
            del obj["exclusiveMinimum"]

        # Recursively process all values
        for value in obj.values():
            _fix_exclusive_minimum_in_schema(value)

    elif isinstance(obj, list):
        # Recursively process all items
        for item in obj:
            _fix_exclusive_minimum_in_schema(item)


def _sort_paths_alphabetically(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Sort the paths in the OpenAPI schema by version prefix first, then alphabetically.
    Also sort HTTP methods alphabetically within each path.
    Version order: v1beta, v1alpha, v1
    """
    if "paths" not in openapi_schema:
        return openapi_schema

    def path_sort_key(path: str) -> tuple:
        """
        Create a sort key that groups paths by version prefix first.
        Returns (version_priority, path) where version_priority:
        - 0 for v1beta
        - 1 for v1alpha
        - 2 for v1
        - 3 for others
        """
        if path.startswith("/v1beta/"):
            version_priority = 0
        elif path.startswith("/v1alpha/"):
            version_priority = 1
        elif path.startswith("/v1/"):
            version_priority = 2
        else:
            version_priority = 3

        return (version_priority, path)

    def sort_path_item(path_item: dict[str, Any]) -> dict[str, Any]:
        """Sort HTTP methods alphabetically within a path item."""
        if not isinstance(path_item, dict):
            return path_item

        # Define the order of HTTP methods
        method_order = ["delete", "get", "head", "options", "patch", "post", "put", "trace"]

        # Create a new ordered dict with methods in alphabetical order
        sorted_path_item = {}

        # First add methods in the defined order
        for method in method_order:
            if method in path_item:
                sorted_path_item[method] = path_item[method]

        # Then add any other keys that aren't HTTP methods
        for key, value in path_item.items():
            if key not in method_order:
                sorted_path_item[key] = value

        return sorted_path_item

    # Sort paths by version prefix first, then alphabetically
    # Also sort HTTP methods within each path
    sorted_paths = {}
    for path, path_item in sorted(openapi_schema["paths"].items(), key=lambda x: path_sort_key(x[0])):
        sorted_paths[path] = sort_path_item(path_item)

    openapi_schema["paths"] = sorted_paths

    return openapi_schema


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

    # Filter paths based on version prefix and deprecated status
    filtered_paths = {}
    for path, path_item in filtered_schema["paths"].items():
        # Check if path has any deprecated operations
        is_deprecated = _is_path_deprecated(path_item)

        # Skip deprecated endpoints if exclude_deprecated is True
        if exclude_deprecated and is_deprecated:
            continue

        if stable_only:
            # Only include /v1/ paths, exclude /v1alpha/ and /v1beta/
            if path.startswith("/v1/") and not path.startswith("/v1alpha/") and not path.startswith("/v1beta/"):
                filtered_paths[path] = path_item
        else:
            # Only include /v1alpha/ and /v1beta/ paths, exclude /v1/
            if path.startswith("/v1alpha/") or path.startswith("/v1beta/"):
                filtered_paths[path] = path_item

    filtered_schema["paths"] = filtered_paths

    # Filter schemas/components to only include ones referenced by filtered paths
    if "components" in filtered_schema and "schemas" in filtered_schema["components"]:
        # Find all schemas that are actually referenced by the filtered paths
        # Use the original schema to find all references, not the filtered one
        referenced_schemas = _find_schemas_referenced_by_paths(filtered_paths, openapi_schema)

        # Only keep schemas that are referenced by the filtered paths
        filtered_schemas = {}
        for schema_name, schema_def in filtered_schema["components"]["schemas"].items():
            if schema_name in referenced_schemas:
                filtered_schemas[schema_name] = schema_def

        filtered_schema["components"]["schemas"] = filtered_schemas

    return filtered_schema


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


def _is_path_deprecated(path_item: dict[str, Any]) -> bool:
    """
    Check if a path item has any deprecated operations.
    """
    if not isinstance(path_item, dict):
        return False

    # Check each HTTP method in the path item
    for method in ["get", "post", "put", "delete", "patch", "head", "options"]:
        if method in path_item:
            operation = path_item[method]
            if isinstance(operation, dict) and operation.get("deprecated", False):
                return True

    return False


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


def generate_openapi_spec(output_dir: str, format: str = "yaml", include_examples: bool = True) -> dict[str, Any]:
    """
    Generate OpenAPI specification using FastAPI's built-in method.

    Args:
        output_dir: Directory to save the generated files
        format: Output format ("yaml", "json", or "both")
        include_examples: Whether to include examples in the spec

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

    # Add Llama Stack specific extensions
    openapi_schema = _add_llama_stack_extensions(openapi_schema, app)

    # Add standard error responses
    openapi_schema = _add_error_responses(openapi_schema)

    # Ensure all referenced schemas are included
    # DISABLED: This was using hardcoded schema generation. FastAPI should handle this automatically.
    # openapi_schema = _ensure_referenced_schemas(openapi_schema)

    # Control schema registration based on @json_schema_type decorator
    # Temporarily disabled to fix missing schema issues
    # openapi_schema = _control_schema_registration(openapi_schema)

    # Fix malformed schemas after all other processing
    # DISABLED: This was a hardcoded workaround. Using Pydantic's TypeAdapter instead.
    # _fix_malformed_schemas(openapi_schema)

    # Split into stable (v1 only), experimental (v1alpha + v1beta), and deprecated specs
    # Each spec needs its own deep copy of the full schema to avoid cross-contamination
    import copy

    stable_schema = _filter_schema_by_version(copy.deepcopy(openapi_schema), stable_only=True, exclude_deprecated=True)
    experimental_schema = _filter_schema_by_version(
        copy.deepcopy(openapi_schema), stable_only=False, exclude_deprecated=True
    )
    deprecated_schema = _filter_deprecated_schema(copy.deepcopy(openapi_schema))

    # Sort paths alphabetically for stable (v1 only)
    stable_schema = _sort_paths_alphabetically(stable_schema)
    # Sort paths by version prefix for experimental (v1beta, v1alpha)
    experimental_schema = _sort_paths_alphabetically(experimental_schema)
    # Sort paths by version prefix for deprecated
    deprecated_schema = _sort_paths_alphabetically(deprecated_schema)

    # Fix schema issues (like exclusiveMinimum -> minimum) for each spec
    stable_schema = _fix_schema_issues(stable_schema)
    experimental_schema = _fix_schema_issues(experimental_schema)
    deprecated_schema = _fix_schema_issues(deprecated_schema)

    # Add any custom modifications here if needed
    if include_examples:
        # Add examples to the schema if needed
        pass

    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save the stable specification
    if format in ["yaml", "both"]:
        yaml_path = output_path / "llama-stack-spec.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(stable_schema, f, default_flow_style=False, sort_keys=False)
        print(f"✅ Generated YAML (stable): {yaml_path}")

        experimental_yaml_path = output_path / "experimental-llama-stack-spec.yaml"
        with open(experimental_yaml_path, "w") as f:
            yaml.dump(experimental_schema, f, default_flow_style=False, sort_keys=False)
        print(f"✅ Generated YAML (experimental): {experimental_yaml_path}")

        deprecated_yaml_path = output_path / "deprecated-llama-stack-spec.yaml"
        with open(deprecated_yaml_path, "w") as f:
            yaml.dump(deprecated_schema, f, default_flow_style=False, sort_keys=False)
        print(f"✅ Generated YAML (deprecated): {deprecated_yaml_path}")

    if format in ["json", "both"]:
        json_path = output_path / "llama-stack-spec.json"
        with open(json_path, "w") as f:
            json.dump(stable_schema, f, indent=2)
        print(f"✅ Generated JSON (stable): {json_path}")

        experimental_json_path = output_path / "experimental-llama-stack-spec.json"
        with open(experimental_json_path, "w") as f:
            json.dump(experimental_schema, f, indent=2)
        print(f"✅ Generated JSON (experimental): {experimental_json_path}")

        deprecated_json_path = output_path / "deprecated-llama-stack-spec.json"
        with open(deprecated_json_path, "w") as f:
            json.dump(deprecated_schema, f, indent=2)
        print(f"✅ Generated JSON (deprecated): {deprecated_json_path}")

    # Generate HTML documentation
    html_path = output_path / "llama-stack-spec.html"
    generate_html_docs(stable_schema, html_path)
    print(f"✅ Generated HTML: {html_path}")

    experimental_html_path = output_path / "experimental-llama-stack-spec.html"
    generate_html_docs(experimental_schema, experimental_html_path, spec_file="experimental-llama-stack-spec.yaml")
    print(f"✅ Generated HTML (experimental): {experimental_html_path}")

    deprecated_html_path = output_path / "deprecated-llama-stack-spec.html"
    generate_html_docs(deprecated_schema, deprecated_html_path, spec_file="deprecated-llama-stack-spec.yaml")
    print(f"✅ Generated HTML (deprecated): {deprecated_html_path}")

    return stable_schema


def generate_html_docs(
    openapi_schema: dict[str, Any], output_path: Path, spec_file: str = "llama-stack-spec.yaml"
) -> None:
    """Generate HTML documentation using ReDoc."""
    html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Llama Stack API Documentation</title>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700" rel="stylesheet">
    <style>
        body {{ margin: 0; padding: 0; }}
    </style>
</head>
<body>
    <redoc spec-url='{spec_file}'></redoc>
    <script src="https://cdn.jsdelivr.net/npm/redoc@2.0.0/bundles/redoc.standalone.js"></script>
</body>
</html>
    """.strip()

    with open(output_path, "w") as f:
        f.write(html_template + "\n")


def main():
    """Main entry point for the FastAPI OpenAPI generator."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate OpenAPI specification using FastAPI")
    parser.add_argument("output_dir", help="Output directory for generated files")
    parser.add_argument("--format", choices=["yaml", "json", "both"], default="yaml", help="Output format")
    parser.add_argument("--no-examples", action="store_true", help="Exclude examples from the specification")

    args = parser.parse_args()

    print("🚀 Generating OpenAPI specification using FastAPI...")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📄 Format: {args.format}")

    try:
        openapi_schema = generate_openapi_spec(
            output_dir=args.output_dir, format=args.format, include_examples=not args.no_examples
        )

        print("\n✅ OpenAPI specification generated successfully!")
        print(f"📊 Schemas: {len(openapi_schema.get('components', {}).get('schemas', {}))}")
        print(f"🛣️  Paths: {len(openapi_schema.get('paths', {}))}")

        # Count operations
        operation_count = 0
        for path_info in openapi_schema.get("paths", {}).values():
            for method in ["get", "post", "put", "delete", "patch"]:
                if method in path_info:
                    operation_count += 1

        print(f"🔧 Operations: {operation_count}")

    except Exception as e:
        print(f"❌ Error generating OpenAPI specification: {e}")
        raise


if __name__ == "__main__":
    main()

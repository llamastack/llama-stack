# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Schema transformations and fixes for OpenAPI generation.
"""

import copy
from collections import OrderedDict
from pathlib import Path
from typing import Any

import yaml
from openapi_spec_validator import validate_spec
from openapi_spec_validator.exceptions import OpenAPISpecValidatorError

from . import endpoints, schema_collection
from ._legacy_order import (
    LEGACY_OPERATION_KEYS,
    LEGACY_PATH_ORDER,
    LEGACY_RESPONSE_ORDER,
    LEGACY_SCHEMA_ORDER,
    LEGACY_SECURITY,
    LEGACY_TAG_GROUPS,
    LEGACY_TAGS,
)
from .state import _extra_body_fields


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


def _normalize_empty_responses(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """Convert empty 200 responses into 204 No Content."""

    for path_item in openapi_schema.get("paths", {}).values():
        if not isinstance(path_item, dict):
            continue
        for method in list(path_item.keys()):
            operation = path_item.get(method)
            if not isinstance(operation, dict):
                continue
            responses = operation.get("responses")
            if not isinstance(responses, dict):
                continue
            response_200 = responses.get("200") or responses.get(200)
            if response_200 is None:
                continue
            content = response_200.get("content")
            if content and any(
                isinstance(media, dict) and media.get("schema") not in ({}, None) for media in content.values()
            ):
                continue
            responses.pop("200", None)
            responses.pop(200, None)
            responses["204"] = {"description": response_200.get("description", "No Content")}
    return openapi_schema


def _eliminate_defs_section(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Eliminate $defs section entirely by moving all definitions to components/schemas.
    This matches the structure of the old pyopenapi generator for oasdiff compatibility.
    """
    schema_collection._ensure_components_schemas(openapi_schema)

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
        from llama_stack_api.datatypes import Error

        schema_collection._ensure_components_schemas(openapi_schema)
        if "Error" not in openapi_schema["components"]["schemas"]:
            openapi_schema["components"]["schemas"]["Error"] = Error.model_json_schema()
    except ImportError:
        pass

    schema_collection._ensure_components_schemas(openapi_schema)
    if "Response" not in openapi_schema["components"]["schemas"]:
        openapi_schema["components"]["schemas"]["Response"] = {"title": "Response", "type": "object"}

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
        path_params = endpoints._extract_path_parameters(path)

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


def _get_schema_title(item: dict[str, Any]) -> str | None:
    """Extract a title for a schema item to use in union variant names."""
    if "$ref" in item:
        return item["$ref"].split("/")[-1]
    elif "type" in item:
        type_val = item["type"]
        if type_val == "null":
            return None
        if type_val == "array" and "items" in item:
            items = item["items"]
            if isinstance(items, dict):
                if "anyOf" in items or "oneOf" in items:
                    nested_union = items.get("anyOf") or items.get("oneOf")
                    if isinstance(nested_union, list) and len(nested_union) > 0:
                        nested_types = []
                        for nested_item in nested_union:
                            if isinstance(nested_item, dict):
                                if "$ref" in nested_item:
                                    nested_types.append(nested_item["$ref"].split("/")[-1])
                                elif "oneOf" in nested_item:
                                    one_of_items = nested_item.get("oneOf", [])
                                    if one_of_items and isinstance(one_of_items[0], dict) and "$ref" in one_of_items[0]:
                                        base_name = one_of_items[0]["$ref"].split("/")[-1].split("-")[0]
                                        nested_types.append(f"{base_name}Union")
                                    else:
                                        nested_types.append("Union")
                                elif "type" in nested_item and nested_item["type"] != "null":
                                    nested_types.append(nested_item["type"])
                        if nested_types:
                            unique_nested = list(dict.fromkeys(nested_types))
                            # Use more descriptive names for better code generation
                            if len(unique_nested) <= 3:
                                return f"list[{' | '.join(unique_nested)}]"
                            else:
                                # Include first few types for better naming
                                return f"list[{unique_nested[0]} | {unique_nested[1]} | ...]"
                        return "list[Union]"
                elif "$ref" in items:
                    return f"list[{items['$ref'].split('/')[-1]}]"
                elif "type" in items:
                    return f"list[{items['type']}]"
            return "array"
        return type_val
    elif "title" in item:
        return item["title"]
    return None


def _add_titles_to_unions(obj: Any, parent_key: str | None = None) -> None:
    """Recursively add titles to union schemas (anyOf/oneOf) to help code generators infer names."""
    if isinstance(obj, dict):
        # Check if this is a union schema (anyOf or oneOf)
        if "anyOf" in obj or "oneOf" in obj:
            union_type = "anyOf" if "anyOf" in obj else "oneOf"
            union_items = obj[union_type]

            if isinstance(union_items, list) and len(union_items) > 0:
                # Skip simple nullable unions (type | null) - these don't need titles
                is_simple_nullable = (
                    len(union_items) == 2
                    and any(isinstance(item, dict) and item.get("type") == "null" for item in union_items)
                    and any(
                        isinstance(item, dict) and "type" in item and item.get("type") != "null" for item in union_items
                    )
                    and not any(
                        isinstance(item, dict) and ("$ref" in item or "anyOf" in item or "oneOf" in item)
                        for item in union_items
                    )
                )

                if is_simple_nullable:
                    # Remove title from simple nullable unions if it exists
                    if "title" in obj:
                        del obj["title"]
                else:
                    # Add titles to individual union variants that need them
                    for item in union_items:
                        if isinstance(item, dict):
                            # Skip null types
                            if item.get("type") == "null":
                                continue
                            # Add title to complex variants (arrays with unions, nested unions, etc.)
                            # Also add to simple types if they're part of a complex union
                            needs_title = (
                                "items" in item
                                or "anyOf" in item
                                or "oneOf" in item
                                or ("$ref" in item and "title" not in item)
                            )
                            if needs_title and "title" not in item:
                                variant_title = _get_schema_title(item)
                                if variant_title:
                                    item["title"] = variant_title

                    # Try to infer a meaningful title from the union items for the parent
                    titles = []
                    for item in union_items:
                        if isinstance(item, dict):
                            title = _get_schema_title(item)
                            if title:
                                titles.append(title)

                    if titles:
                        # Create a title from the union items
                        unique_titles = list(dict.fromkeys(titles))  # Preserve order, remove duplicates
                        if len(unique_titles) <= 3:
                            title = " | ".join(unique_titles)
                        else:
                            title = f"{unique_titles[0]} | ... ({len(unique_titles)} variants)"
                        # Always set the title for unions to help code generators
                        # This will replace generic property titles with union-specific ones
                        obj["title"] = title
                    elif "title" not in obj and parent_key:
                        # Use parent key as fallback only if no title exists
                        obj["title"] = f"{parent_key.title()}Union"

        # Recursively process all values
        for key, value in obj.items():
            _add_titles_to_unions(value, key)
    elif isinstance(obj, list):
        for item in obj:
            _add_titles_to_unions(item, parent_key)


def _convert_anyof_const_to_enum(obj: Any) -> None:
    """Convert anyOf with multiple const string values to a proper enum."""
    if isinstance(obj, dict):
        if "anyOf" in obj:
            any_of = obj["anyOf"]
            if isinstance(any_of, list):
                # Check if all items are const string values
                const_values = []
                has_null = False
                can_convert = True
                for item in any_of:
                    if isinstance(item, dict):
                        if item.get("type") == "null":
                            has_null = True
                        elif item.get("type") == "string" and "const" in item:
                            const_values.append(item["const"])
                        else:
                            # Not a simple const pattern, skip conversion for this anyOf
                            can_convert = False
                            break

                # If we have const values and they're all strings, convert to enum
                if can_convert and const_values and len(const_values) == len(any_of) - (1 if has_null else 0):
                    # Convert to enum
                    obj["type"] = "string"
                    obj["enum"] = const_values
                    # Preserve default if present, otherwise try to get from first const item
                    if "default" not in obj:
                        for item in any_of:
                            if isinstance(item, dict) and "const" in item:
                                obj["default"] = item["const"]
                                break
                    # Remove anyOf
                    del obj["anyOf"]
                    # Handle nullable
                    if has_null:
                        obj["nullable"] = True
                    # Remove title if it's just "string"
                    if obj.get("title") == "string":
                        del obj["title"]

        # Recursively process all values
        for value in obj.values():
            _convert_anyof_const_to_enum(value)
    elif isinstance(obj, list):
        for item in obj:
            _convert_anyof_const_to_enum(item)


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


def _extract_duplicate_union_types(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Extract duplicate union types to shared schema references.

    Stainless generates type names from union types based on their context, which can cause
    duplicate names when the same union appears in different places. This function extracts
    these duplicate unions to shared schema definitions and replaces inline definitions with
    references to them.

    According to Stainless docs, when duplicate types are detected, they should be extracted
    to the same ref and declared as a model. This ensures Stainless generates consistent
    type names regardless of where the union is referenced.

    Fixes: https://www.stainless.com/docs/reference/diagnostics#Python/DuplicateDeclaration
    """
    if "components" not in openapi_schema or "schemas" not in openapi_schema["components"]:
        return openapi_schema

    schemas = openapi_schema["components"]["schemas"]

    # Extract the Output union type (used in OpenAIResponseObjectWithInput-Output and ListOpenAIResponseInputItem)
    output_union_schema_name = "OpenAIResponseMessageOutputUnion"
    output_union_title = None

    # Get the union type from OpenAIResponseObjectWithInput-Output.input.items.anyOf
    if "OpenAIResponseObjectWithInput-Output" in schemas:
        schema = schemas["OpenAIResponseObjectWithInput-Output"]
        if isinstance(schema, dict) and "properties" in schema:
            input_prop = schema["properties"].get("input")
            if isinstance(input_prop, dict) and "items" in input_prop:
                items = input_prop["items"]
                if isinstance(items, dict) and "anyOf" in items:
                    # Extract the union schema with deep copy
                    output_union_schema = copy.deepcopy(items["anyOf"])
                    output_union_title = items.get("title", "OpenAIResponseMessageOutputUnion")

                    # Collect all refs from the oneOf to detect duplicates
                    refs_in_oneof = set()
                    for item in output_union_schema:
                        if isinstance(item, dict) and "oneOf" in item:
                            oneof = item["oneOf"]
                            if isinstance(oneof, list):
                                for variant in oneof:
                                    if isinstance(variant, dict) and "$ref" in variant:
                                        refs_in_oneof.add(variant["$ref"])
                            item["x-stainless-naming"] = "OpenAIResponseMessageOutputOneOf"

                    # Remove duplicate refs from anyOf that are already in oneOf
                    deduplicated_schema = []
                    for item in output_union_schema:
                        if isinstance(item, dict) and "$ref" in item:
                            if item["$ref"] not in refs_in_oneof:
                                deduplicated_schema.append(item)
                        else:
                            deduplicated_schema.append(item)
                    output_union_schema = deduplicated_schema

                    # Create the shared schema with x-stainless-naming to ensure consistent naming
                    if output_union_schema_name not in schemas:
                        schemas[output_union_schema_name] = {
                            "anyOf": output_union_schema,
                            "title": output_union_title,
                            "x-stainless-naming": output_union_schema_name,
                        }
                    # Replace with reference
                    input_prop["items"] = {"$ref": f"#/components/schemas/{output_union_schema_name}"}

    # Replace the same union in ListOpenAIResponseInputItem.data.items.anyOf
    if "ListOpenAIResponseInputItem" in schemas and output_union_schema_name in schemas:
        schema = schemas["ListOpenAIResponseInputItem"]
        if isinstance(schema, dict) and "properties" in schema:
            data_prop = schema["properties"].get("data")
            if isinstance(data_prop, dict) and "items" in data_prop:
                items = data_prop["items"]
                if isinstance(items, dict) and "anyOf" in items:
                    # Replace with reference
                    data_prop["items"] = {"$ref": f"#/components/schemas/{output_union_schema_name}"}

    # Extract the Input union type (used in _responses_Request.input.anyOf[1].items.anyOf)
    input_union_schema_name = "OpenAIResponseMessageInputUnion"

    if "_responses_Request" in schemas:
        schema = schemas["_responses_Request"]
        if isinstance(schema, dict) and "properties" in schema:
            input_prop = schema["properties"].get("input")
            if isinstance(input_prop, dict) and "anyOf" in input_prop:
                any_of = input_prop["anyOf"]
                if isinstance(any_of, list) and len(any_of) > 1:
                    # Check the second item (index 1) which should be the array type
                    second_item = any_of[1]
                    if isinstance(second_item, dict) and "items" in second_item:
                        items = second_item["items"]
                        if isinstance(items, dict) and "anyOf" in items:
                            # Extract the union schema with deep copy
                            input_union_schema = copy.deepcopy(items["anyOf"])
                            input_union_title = items.get("title", "OpenAIResponseMessageInputUnion")

                            # Collect all refs from the oneOf to detect duplicates
                            refs_in_oneof = set()
                            for item in input_union_schema:
                                if isinstance(item, dict) and "oneOf" in item:
                                    oneof = item["oneOf"]
                                    if isinstance(oneof, list):
                                        for variant in oneof:
                                            if isinstance(variant, dict) and "$ref" in variant:
                                                refs_in_oneof.add(variant["$ref"])
                                    item["x-stainless-naming"] = "OpenAIResponseMessageInputOneOf"

                            # Remove duplicate refs from anyOf that are already in oneOf
                            deduplicated_schema = []
                            for item in input_union_schema:
                                if isinstance(item, dict) and "$ref" in item:
                                    if item["$ref"] not in refs_in_oneof:
                                        deduplicated_schema.append(item)
                                else:
                                    deduplicated_schema.append(item)
                            input_union_schema = deduplicated_schema

                            # Create the shared schema with x-stainless-naming to ensure consistent naming
                            if input_union_schema_name not in schemas:
                                schemas[input_union_schema_name] = {
                                    "anyOf": input_union_schema,
                                    "title": input_union_title,
                                    "x-stainless-naming": input_union_schema_name,
                                }
                            # Replace with reference
                            second_item["items"] = {"$ref": f"#/components/schemas/{input_union_schema_name}"}

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

    # Post-process to remove trailing whitespace from all lines
    with open(file_path) as f:
        lines = f.readlines()

    # Strip trailing whitespace from each line, preserving newlines
    cleaned_lines = [line.rstrip() + "\n" if line.endswith("\n") else line.rstrip() for line in lines]

    with open(file_path, "w") as f:
        f.writelines(cleaned_lines)


def _apply_legacy_sorting(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Temporarily match the legacy ordering from origin/main so diffs are easier to read.
    Remove this once the generator output stabilizes and we no longer need legacy diffs.
    """

    def order_mapping(data: dict[str, Any], priority: list[str]) -> OrderedDict[str, Any]:
        ordered: OrderedDict[str, Any] = OrderedDict()
        for key in priority:
            if key in data:
                ordered[key] = data[key]
        for key, value in data.items():
            if key not in ordered:
                ordered[key] = value
        return ordered

    paths = openapi_schema.get("paths")
    if isinstance(paths, dict):
        openapi_schema["paths"] = order_mapping(paths, LEGACY_PATH_ORDER)
        for path, path_item in openapi_schema["paths"].items():
            if not isinstance(path_item, dict):
                continue
            ordered_path_item = OrderedDict()
            for method in ["get", "post", "put", "delete", "patch", "head", "options"]:
                if method in path_item:
                    ordered_path_item[method] = order_mapping(path_item[method], LEGACY_OPERATION_KEYS)
            for key, value in path_item.items():
                if key not in ordered_path_item:
                    if isinstance(value, dict) and key.lower() in {
                        "get",
                        "post",
                        "put",
                        "delete",
                        "patch",
                        "head",
                        "options",
                    }:
                        ordered_path_item[key] = order_mapping(value, LEGACY_OPERATION_KEYS)
                    else:
                        ordered_path_item[key] = value
            openapi_schema["paths"][path] = ordered_path_item

    components = openapi_schema.setdefault("components", {})
    schemas = components.get("schemas")
    if isinstance(schemas, dict):
        components["schemas"] = order_mapping(schemas, LEGACY_SCHEMA_ORDER)
    responses = components.get("responses")
    if isinstance(responses, dict):
        components["responses"] = order_mapping(responses, LEGACY_RESPONSE_ORDER)

    if LEGACY_TAGS:
        openapi_schema["tags"] = LEGACY_TAGS

    if LEGACY_TAG_GROUPS:
        openapi_schema["x-tagGroups"] = LEGACY_TAG_GROUPS

    if LEGACY_SECURITY:
        openapi_schema["security"] = LEGACY_SECURITY

    return openapi_schema


def _fix_schema_issues(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """Fix common schema issues: exclusiveMinimum, null defaults, and add titles to unions."""
    # Convert anyOf with const values to enums across the entire schema
    _convert_anyof_const_to_enum(openapi_schema)

    # Fix other schema issues and add titles to unions
    if "components" in openapi_schema and "schemas" in openapi_schema["components"]:
        for schema_name, schema_def in openapi_schema["components"]["schemas"].items():
            _fix_schema_recursive(schema_def)
            _add_titles_to_unions(schema_def, schema_name)
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
        print(f"{schema_name} is valid")
        return True
    except OpenAPISpecValidatorError as e:
        print(f"{schema_name} validation failed: {e}")
        return False
    except Exception as e:
        print(f"{schema_name} validation error: {e}")
        return False

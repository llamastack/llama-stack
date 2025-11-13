# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Schema filtering and version filtering for OpenAPI generation.
"""

from typing import Any

from llama_stack.apis.version import (
    LLAMA_STACK_API_V1,
    LLAMA_STACK_API_V1ALPHA,
    LLAMA_STACK_API_V1BETA,
)

from . import schema_collection


def _get_all_json_schema_type_names() -> set[str]:
    """
    Get all schema names from @json_schema_type decorated models.
    This ensures they are included in filtered schemas even if not directly referenced by paths.
    """
    schema_names = set()
    apis_modules = schema_collection._import_all_modules_in_package("llama_stack.apis")
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


def _get_explicit_schema_names(openapi_schema: dict[str, Any]) -> set[str]:
    """Get all registered schema names and @json_schema_type decorated model names."""
    from llama_stack.schema_utils import _registered_schemas

    registered_schema_names = {info["name"] for info in _registered_schemas.values()}
    json_schema_type_names = _get_all_json_schema_type_names()
    return registered_schema_names | json_schema_type_names


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


def _is_path_deprecated(path_item: dict[str, Any]) -> bool:
    """Check if a path item has any deprecated operations."""
    if not isinstance(path_item, dict):
        return False
    for method in ["get", "post", "put", "delete", "patch", "head", "options"]:
        if isinstance(path_item.get(method), dict) and path_item[method].get("deprecated", False):
            return True
    return False


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
    Excludes deprecated endpoints. This is used for the combined "stainless" spec.
    """
    filtered_schema = openapi_schema.copy()

    if "paths" not in filtered_schema:
        return filtered_schema

    # Filter paths to include stable (v1) and experimental (v1alpha, v1beta), excluding deprecated
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

            # Skip deprecated operations
            if operation.get("deprecated", False):
                continue

            filtered_path_item[method] = operation

        # Only include path if it has at least one operation after filtering
        if filtered_path_item:
            # Check if path matches version filter (stable or experimental)
            if _is_stable_path(path) or _is_experimental_path(path):
                filtered_paths[path] = filtered_path_item

    filtered_schema["paths"] = filtered_paths

    return _filter_schemas_by_references(filtered_schema, filtered_paths, openapi_schema)

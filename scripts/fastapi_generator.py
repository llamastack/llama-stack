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
import json
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from openapi_spec_validator import validate_spec
from openapi_spec_validator.exceptions import OpenAPISpecValidatorError

from llama_stack.apis.version import LLAMA_STACK_API_V1, LLAMA_STACK_API_V1ALPHA, LLAMA_STACK_API_V1BETA
from llama_stack.core.distribution import INTERNAL_APIS, providable_apis
from llama_stack.core.resolver import api_protocol_map


def create_llama_stack_app() -> FastAPI:
    """
    Create a FastAPI app that represents the Llama Stack API.
    All APIs use FastAPI routers for OpenAPI generation.
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

    # Import API modules to ensure routers are registered (they register on import)
    # Import all providable APIs plus internal APIs that need routers
    apis_to_import = set(providable_apis()) | INTERNAL_APIS

    # Map API enum values to their actual module names (for APIs where they differ)
    api_module_map = {
        "tool_runtime": "tools",
        "tool_groups": "tools",
    }

    imported_modules = set()
    for api in apis_to_import:
        module_name = api_module_map.get(api.value, api.value)  # type: ignore[attr-defined]

        # Skip if we've already imported this module (e.g., both tool_runtime and tool_groups use 'tools')
        if module_name in imported_modules:
            continue

        try:
            importlib.import_module(f"llama_stack.apis.{module_name}")
            imported_modules.add(module_name)
        except ImportError:
            print(
                f"❌ Failed to import module {module_name}, this API will not be included in the OpenAPI specification"
            )
            pass

    # Import router registry
    from llama_stack.core.server.routers import create_router, has_router
    from llama_stack.providers.datatypes import Api

    # Get all APIs that should be served
    protocols = api_protocol_map()
    apis_to_serve = set(protocols.keys())

    # Create a dummy impl_getter that returns a mock implementation
    # This is only for OpenAPI generation, so we don't need real implementations
    class MockImpl:
        pass

    def impl_getter(api: Api) -> Any:
        return MockImpl()

    # Register all routers - all APIs now use routers
    for api in apis_to_serve:
        if has_router(api):
            router = create_router(api, impl_getter)
            if router:
                app.include_router(router)

    return app


def _ensure_json_schema_types_included(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Ensure all @json_schema_type decorated models are included in the OpenAPI schema.
    This finds all models with the _llama_stack_schema_type attribute and adds them to the schema.
    """
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}

    if "schemas" not in openapi_schema["components"]:
        openapi_schema["components"]["schemas"] = {}

    # Find all classes with the _llama_stack_schema_type attribute
    from llama_stack import apis

    # Get all modules in the apis package
    apis_modules = []
    for module_name in dir(apis):
        if not module_name.startswith("_"):
            try:
                module = getattr(apis, module_name)
                if hasattr(module, "__file__"):
                    apis_modules.append(module)
            except (ImportError, AttributeError):
                continue

    # Also check submodules
    for module in apis_modules:
        for attr_name in dir(module):
            if not attr_name.startswith("_"):
                try:
                    attr = getattr(module, attr_name)
                    if hasattr(attr, "__file__") and hasattr(attr, "__name__"):
                        apis_modules.append(attr)
                except (ImportError, AttributeError):
                    continue

    # Find all classes with the _llama_stack_schema_type attribute
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
                            schema = attr.model_json_schema()
                            openapi_schema["components"]["schemas"][schema_name] = schema
                        except Exception:
                            # Skip if we can't generate the schema
                            continue
            except (AttributeError, TypeError):
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


def _fix_schema_issues(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Fix common schema issues that cause OpenAPI validation problems.
    This includes converting exclusiveMinimum numbers to minimum values and fixing string fields with null defaults.
    """
    if "components" not in openapi_schema or "schemas" not in openapi_schema["components"]:
        return openapi_schema

    schemas = openapi_schema["components"]["schemas"]

    # Fix exclusiveMinimum issues
    for _, schema_def in schemas.items():
        _fix_exclusive_minimum_in_schema(schema_def)

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


def _get_path_version(path: str) -> str | None:
    """
    Determine the API version of a path based on its prefix.

    Args:
        path: The API path (e.g., "/v1/datasets", "/v1beta/models")

    Returns:
        Version string ("v1", "v1alpha", "v1beta") or None if no recognized version
    """
    if path.startswith("/" + LLAMA_STACK_API_V1BETA):
        return "v1beta"
    elif path.startswith("/" + LLAMA_STACK_API_V1ALPHA):
        return "v1alpha"
    elif path.startswith("/" + LLAMA_STACK_API_V1):
        return "v1"
    return None


def _is_stable_path(path: str) -> bool:
    """Check if a path is a stable v1 path (not experimental)."""
    return (
        path.startswith("/" + LLAMA_STACK_API_V1)
        and not path.startswith("/" + LLAMA_STACK_API_V1ALPHA)
        and not path.startswith("/" + LLAMA_STACK_API_V1BETA)
    )


def _is_experimental_path(path: str) -> bool:
    """Check if a path is experimental (v1alpha or v1beta)."""
    return path.startswith("/" + LLAMA_STACK_API_V1ALPHA) or path.startswith("/" + LLAMA_STACK_API_V1BETA)


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
        version = _get_path_version(path)
        version_priority_map = {LLAMA_STACK_API_V1BETA: 0, LLAMA_STACK_API_V1ALPHA: 1, LLAMA_STACK_API_V1: 2}
        version_priority = version_priority_map.get(version, 3) if version else 3
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


def _should_include_path(
    path: str, path_item: dict[str, Any], include_stable: bool, include_experimental: bool, exclude_deprecated: bool
) -> bool:
    """
    Determine if a path should be included in the filtered schema.

    Args:
        path: The API path
        path_item: The path item from OpenAPI schema
        include_stable: Whether to include stable v1 paths
        include_experimental: Whether to include experimental (v1alpha/v1beta) paths
        exclude_deprecated: Whether to exclude deprecated endpoints

    Returns:
        True if the path should be included
    """
    if exclude_deprecated and _is_path_deprecated(path_item):
        return False

    is_stable = _is_stable_path(path)
    is_experimental = _is_experimental_path(path)

    if is_stable and include_stable:
        return True
    if is_experimental and include_experimental:
        return True

    return False


def _filter_schema(
    openapi_schema: dict[str, Any],
    include_stable: bool = True,
    include_experimental: bool = False,
    deprecated_mode: str = "exclude",
    filter_schemas: bool = True,
) -> dict[str, Any]:
    """
    Filter OpenAPI schema by version and deprecated status.

    Args:
        openapi_schema: The full OpenAPI schema
        include_stable: Whether to include stable v1 paths
        include_experimental: Whether to include experimental (v1alpha/v1beta) paths
        deprecated_mode: One of "include", "exclude", or "only"
        filter_schemas: Whether to filter components/schemas to only referenced ones

    Returns:
        Filtered OpenAPI schema
    """
    filtered_schema = openapi_schema.copy()

    if "paths" not in filtered_schema:
        return filtered_schema

    # Determine deprecated filtering logic
    if deprecated_mode == "only":
        exclude_deprecated = False
        include_deprecated_only = True
    elif deprecated_mode == "exclude":
        exclude_deprecated = True
        include_deprecated_only = False
    else:  # "include"
        exclude_deprecated = False
        include_deprecated_only = False

    # Filter paths
    filtered_paths = {}
    for path, path_item in filtered_schema["paths"].items():
        is_deprecated = _is_path_deprecated(path_item)

        if include_deprecated_only:
            if is_deprecated:
                filtered_paths[path] = path_item
        elif _should_include_path(path, path_item, include_stable, include_experimental, exclude_deprecated):
            filtered_paths[path] = path_item

    filtered_schema["paths"] = filtered_paths

    # Filter schemas/components if requested
    if filter_schemas and "components" in filtered_schema and "schemas" in filtered_schema["components"]:
        referenced_schemas = _find_schemas_referenced_by_paths(filtered_paths, openapi_schema)
        filtered_schema["components"]["schemas"] = {
            name: schema
            for name, schema in filtered_schema["components"]["schemas"].items()
            if name in referenced_schemas
        }

    # Preserve $defs section if it exists
    if "components" in openapi_schema and "$defs" in openapi_schema["components"]:
        if "components" not in filtered_schema:
            filtered_schema["components"] = {}
        filtered_schema["components"]["$defs"] = openapi_schema["components"]["$defs"]

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

    # Ensure all @json_schema_type decorated models are included
    openapi_schema = _ensure_json_schema_types_included(openapi_schema)

    # Fix $ref references to point to components/schemas instead of $defs
    openapi_schema = _fix_ref_references(openapi_schema)

    # Split into stable (v1 only), experimental (v1alpha + v1beta), deprecated, and combined (stainless) specs
    # Each spec needs its own deep copy of the full schema to avoid cross-contamination
    import copy

    stable_schema = _filter_schema(
        copy.deepcopy(openapi_schema), include_stable=True, include_experimental=False, deprecated_mode="exclude"
    )
    experimental_schema = _filter_schema(
        copy.deepcopy(openapi_schema), include_stable=False, include_experimental=True, deprecated_mode="exclude"
    )
    deprecated_schema = _filter_schema(
        copy.deepcopy(openapi_schema),
        include_stable=True,
        include_experimental=True,
        deprecated_mode="only",
        filter_schemas=False,
    )
    combined_schema = _filter_schema(
        copy.deepcopy(openapi_schema), include_stable=True, include_experimental=True, deprecated_mode="exclude"
    )

    # Update title and description for combined schema
    if "info" in combined_schema:
        combined_schema["info"]["title"] = "Llama Stack API - Stable & Experimental APIs"
        combined_schema["info"]["description"] = (
            combined_schema["info"].get("description", "")
            + "\n\n**🔗 COMBINED**: This specification includes both stable production-ready APIs and experimental pre-release APIs. "
            "Use stable APIs for production deployments and experimental APIs for testing new features."
        )

    # Sort paths alphabetically for stable (v1 only)
    stable_schema = _sort_paths_alphabetically(stable_schema)
    # Sort paths by version prefix for experimental (v1beta, v1alpha)
    experimental_schema = _sort_paths_alphabetically(experimental_schema)
    # Sort paths by version prefix for deprecated
    deprecated_schema = _sort_paths_alphabetically(deprecated_schema)
    # Sort paths by version prefix for combined (stainless)
    combined_schema = _sort_paths_alphabetically(combined_schema)

    # Fix schema issues (like exclusiveMinimum -> minimum) for each spec
    stable_schema = _fix_schema_issues(stable_schema)
    experimental_schema = _fix_schema_issues(experimental_schema)
    deprecated_schema = _fix_schema_issues(deprecated_schema)
    combined_schema = _fix_schema_issues(combined_schema)

    # Validate the schemas
    validate_openapi_schema(stable_schema, "Stable schema")
    validate_openapi_schema(experimental_schema, "Experimental schema")
    validate_openapi_schema(deprecated_schema, "Deprecated schema")
    validate_openapi_schema(combined_schema, "Combined (stainless) schema")

    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save the stable specification
    yaml_path = output_path / "llama-stack-spec.yaml"

    # Use ruamel.yaml for better YAML formatting
    try:
        from ruamel.yaml import YAML

        yaml_writer = YAML()
        yaml_writer.default_flow_style = False
        yaml_writer.sort_keys = False
        yaml_writer.width = 4096  # Prevent line wrapping
        yaml_writer.allow_unicode = True

        with open(yaml_path, "w") as f:
            yaml_writer.dump(stable_schema, f)

        # Post-process the YAML file to remove $defs section and fix references
        # Re-read and re-write with ruamel.yaml
        with open(yaml_path) as f:
            yaml_content = f.read()

        if "#/$defs/" in yaml_content:
            yaml_content = yaml_content.replace("#/$defs/", "#/components/schemas/")
            import yaml as pyyaml

            with open(yaml_path) as f:
                yaml_data = pyyaml.safe_load(f)

            if "$defs" in yaml_data:
                if "components" not in yaml_data:
                    yaml_data["components"] = {}
                if "schemas" not in yaml_data["components"]:
                    yaml_data["components"]["schemas"] = {}
                yaml_data["components"]["schemas"].update(yaml_data["$defs"])
                del yaml_data["$defs"]

            with open(yaml_path, "w") as f:
                yaml_writer.dump(yaml_data, f)
    except ImportError:
        # Fallback to standard yaml if ruamel.yaml is not available
        with open(yaml_path, "w") as f:
            yaml.dump(stable_schema, f, default_flow_style=False, sort_keys=False)

    for name, schema in [
        ("experimental", experimental_schema),
        ("deprecated", deprecated_schema),
        ("stainless", combined_schema),
    ]:
        file_path = output_path / f"{name}-llama-stack-spec.yaml"
        try:
            from ruamel.yaml import YAML

            yaml_writer = YAML()
            yaml_writer.default_flow_style = False
            yaml_writer.sort_keys = False
            yaml_writer.width = 4096
            yaml_writer.allow_unicode = True
            with open(file_path, "w") as f:
                yaml_writer.dump(schema, f)
        except ImportError:
            with open(file_path, "w") as f:
                yaml.dump(schema, f, default_flow_style=False, sort_keys=False)

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

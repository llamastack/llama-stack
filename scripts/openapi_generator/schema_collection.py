# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Schema discovery and collection for OpenAPI generation.
"""

import importlib
import pkgutil
from typing import Any

from .state import _dynamic_models


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
        package_name: The fully qualified package name (e.g., 'llama_stack_api')

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
                # (e.g., llama_stack_api.scoring_functions.scoring_functions)
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
    apis_modules = _import_all_modules_in_package("llama_stack_api")
    _import_all_modules_in_package("llama_stack.core.telemetry")

    # First, handle registered schemas (union types, etc.)
    from llama_stack_api.schema_utils import _registered_schemas

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

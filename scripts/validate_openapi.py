#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
OpenAPI Schema Validator for Llama Stack.

This script provides comprehensive validation of OpenAPI specifications
using multiple validation tools and approaches.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import yaml
from openapi_spec_validator import validate_spec
from openapi_spec_validator.exceptions import OpenAPISpecValidatorError


def validate_openapi_schema(schema: dict[str, Any], schema_name: str = "OpenAPI schema") -> bool:
    """
    Validate an OpenAPI schema using openapi-spec-validator.

    Args:
        schema: The OpenAPI schema dictionary to validate
        schema_name: Name of the schema for error reporting

    Returns:
        True if valid, False otherwise
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
        print("   Traceback:")
        traceback.print_exc()
        return False


def validate_schema_file(file_path: Path) -> bool:
    """
    Validate an OpenAPI schema file (YAML or JSON).

    Args:
        file_path: Path to the schema file

    Returns:
        True if valid, False otherwise
    """
    try:
        with open(file_path) as f:
            if file_path.suffix.lower() in [".yaml", ".yml"]:
                schema = yaml.safe_load(f)
            elif file_path.suffix.lower() == ".json":
                schema = json.load(f)
            else:
                print(f"❌ Unsupported file format: {file_path.suffix}")
                return False

        return validate_openapi_schema(schema, str(file_path))
    except Exception as e:
        print(f"❌ Failed to read {file_path}: {e}")
        return False


def validate_directory(directory: Path, pattern: str = "*.yaml") -> bool:
    """
    Validate all OpenAPI schema files in a directory.

    Args:
        directory: Directory containing schema files
        pattern: Glob pattern to match schema files

    Returns:
        True if all files are valid, False otherwise
    """
    if not directory.exists():
        print(f"❌ Directory not found: {directory}")
        return False

    schema_files = list(directory.glob(pattern)) + list(directory.glob("*.yml")) + list(directory.glob("*.json"))

    if not schema_files:
        print(f"❌ No schema files found in {directory}")
        return False

    print(f"🔍 Found {len(schema_files)} schema files to validate")

    all_valid = True
    for schema_file in schema_files:
        print(f"\n📄 Validating {schema_file.name}...")
        is_valid = validate_schema_file(schema_file)
        if not is_valid:
            all_valid = False

    return all_valid


def get_schema_stats(schema: dict[str, Any]) -> dict[str, int]:
    """
    Get statistics about an OpenAPI schema.

    Args:
        schema: The OpenAPI schema dictionary

    Returns:
        Dictionary with schema statistics
    """
    stats = {
        "paths": len(schema.get("paths", {})),
        "schemas": len(schema.get("components", {}).get("schemas", {})),
        "operations": 0,
        "parameters": 0,
        "responses": 0,
    }

    # Count operations
    for path_info in schema.get("paths", {}).values():
        for method in ["get", "post", "put", "delete", "patch", "head", "options"]:
            if method in path_info:
                stats["operations"] += 1

                operation = path_info[method]
                if "parameters" in operation:
                    stats["parameters"] += len(operation["parameters"])
                if "responses" in operation:
                    stats["responses"] += len(operation["responses"])

    return stats


def print_schema_stats(schema: dict[str, Any], schema_name: str = "Schema") -> None:
    """
    Print statistics about an OpenAPI schema.

    Args:
        schema: The OpenAPI schema dictionary
        schema_name: Name of the schema for display
    """
    stats = get_schema_stats(schema)

    print(f"\n📊 {schema_name} Statistics:")
    print(f"   🛣️  Paths: {stats['paths']}")
    print(f"   📋 Schemas: {stats['schemas']}")
    print(f"   🔧 Operations: {stats['operations']}")
    print(f"   📝 Parameters: {stats['parameters']}")
    print(f"   📤 Responses: {stats['responses']}")


def main():
    """Main entry point for the OpenAPI validator."""
    parser = argparse.ArgumentParser(
        description="Validate OpenAPI specifications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a specific file
  python validate_openapi.py docs/static/llama-stack-spec.yaml

  # Validate all YAML files in a directory
  python validate_openapi.py docs/static/

  # Validate with detailed statistics
  python validate_openapi.py docs/static/llama-stack-spec.yaml --stats

  # Validate and show only errors
  python validate_openapi.py docs/static/ --quiet
        """,
    )

    parser.add_argument("path", help="Path to schema file or directory containing schema files")
    parser.add_argument("--stats", action="store_true", help="Show detailed schema statistics")
    parser.add_argument("--quiet", action="store_true", help="Only show errors, suppress success messages")
    parser.add_argument("--pattern", default="*.yaml", help="Glob pattern for schema files (default: *.yaml)")

    args = parser.parse_args()

    path = Path(args.path)

    if not path.exists():
        print(f"❌ Path not found: {path}")
        return 1

    if path.is_file():
        # Validate a single file
        if args.quiet:
            # Override the validation function to be quiet
            def quiet_validate(schema, name):
                try:
                    validate_spec(schema)
                    return True
                except Exception as e:
                    print(f"❌ {name}: {e}")
                    return False

            try:
                with open(path) as f:
                    if path.suffix.lower() in [".yaml", ".yml"]:
                        schema = yaml.safe_load(f)
                    elif path.suffix.lower() == ".json":
                        schema = json.load(f)
                    else:
                        print(f"❌ Unsupported file format: {path.suffix}")
                        return 1

                is_valid = quiet_validate(schema, str(path))
                if is_valid and args.stats:
                    print_schema_stats(schema, path.name)
                return 0 if is_valid else 1
            except Exception as e:
                print(f"❌ Failed to read {path}: {e}")
                return 1
        else:
            is_valid = validate_schema_file(path)
            if is_valid and args.stats:
                try:
                    with open(path) as f:
                        if path.suffix.lower() in [".yaml", ".yml"]:
                            schema = yaml.safe_load(f)
                        elif path.suffix.lower() == ".json":
                            schema = json.load(f)
                        else:
                            return 1
                    print_schema_stats(schema, path.name)
                except Exception:
                    pass
            return 0 if is_valid else 1

    elif path.is_dir():
        # Validate all files in directory
        if args.quiet:
            all_valid = True
            schema_files = list(path.glob(args.pattern)) + list(path.glob("*.yml")) + list(path.glob("*.json"))

            for schema_file in schema_files:
                try:
                    with open(schema_file) as f:
                        if schema_file.suffix.lower() in [".yaml", ".yml"]:
                            schema = yaml.safe_load(f)
                        elif schema_file.suffix.lower() == ".json":
                            schema = json.load(f)
                        else:
                            continue

                    try:
                        validate_spec(schema)
                    except Exception as e:
                        print(f"❌ {schema_file.name}: {e}")
                        all_valid = False
                except Exception as e:
                    print(f"❌ Failed to read {schema_file.name}: {e}")
                    all_valid = False

            return 0 if all_valid else 1
        else:
            all_valid = validate_directory(path, args.pattern)
            if all_valid and args.stats:
                # Show stats for all files
                schema_files = list(path.glob(args.pattern)) + list(path.glob("*.yml")) + list(path.glob("*.json"))
                for schema_file in schema_files:
                    try:
                        with open(schema_file) as f:
                            if schema_file.suffix.lower() in [".yaml", ".yml"]:
                                schema = yaml.safe_load(f)
                            elif schema_file.suffix.lower() == ".json":
                                schema = json.load(f)
                            else:
                                continue
                        print_schema_stats(schema, schema_file.name)
                    except Exception:
                        continue
            return 0 if all_valid else 1

    else:
        print(f"❌ Invalid path type: {path}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

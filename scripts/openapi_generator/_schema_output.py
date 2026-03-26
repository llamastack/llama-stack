# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Output-stage schema transforms: Stainless deduplication, YAML serialization,
and legacy sorting for diff stability.
"""

from collections import OrderedDict
from pathlib import Path
from typing import Any

import yaml

from ._legacy_order import (
    LEGACY_OPERATION_KEYS,
    LEGACY_PATH_ORDER,
    LEGACY_RESPONSE_ORDER,
    LEGACY_SCHEMA_ORDER,
    LEGACY_SECURITY,
    LEGACY_TAG_GROUPS,
    LEGACY_TAGS,
)


def _dedupe_create_response_request_input_union_for_stainless(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """Deduplicate inline unions in `CreateResponseRequest.input` for Stainless.

    The stable OpenAPI spec intentionally preserves legacy structure for oasdiff
    compatibility, but Stainless codegen treats duplicated inline unions as separate
    types and can generate name clashes.

    This transform is intended to run only on the combined (stainless) spec.
    """
    if "components" not in openapi_schema or "schemas" not in openapi_schema["components"]:
        return openapi_schema

    schemas = openapi_schema["components"]["schemas"]
    create_request = schemas.get("CreateResponseRequest")
    if not isinstance(create_request, dict):
        return openapi_schema

    properties = create_request.get("properties")
    if not isinstance(properties, dict):
        return openapi_schema

    input_prop = properties.get("input")
    if not isinstance(input_prop, dict):
        return openapi_schema

    any_of = input_prop.get("anyOf")
    if not isinstance(any_of, list):
        return openapi_schema

    array_schema: dict[str, Any] | None = None
    for item in any_of:
        if isinstance(item, dict) and (item.get("type") == "array" or "items" in item):
            array_schema = item
            break

    if array_schema is None:
        return openapi_schema

    items_schema = array_schema.get("items")
    if not isinstance(items_schema, dict):
        return openapi_schema

    items_any_of = items_schema.get("anyOf")
    if not isinstance(items_any_of, list):
        return openapi_schema

    def _collect_refs(obj: Any, refs: set[str]) -> None:
        if isinstance(obj, dict):
            ref = obj.get("$ref")
            if isinstance(ref, str):
                refs.add(ref)
            for value in obj.values():
                _collect_refs(value, refs)
        elif isinstance(obj, list):
            for value in obj:
                _collect_refs(value, refs)

    def _is_direct_ref_item(item: dict[str, Any]) -> bool:
        if "$ref" not in item:
            return False
        # Direct refs are the sibling entries like: {$ref: ..., title: ...}
        # Union/nesting containers also include oneOf/anyOf/items/etc.
        container_keys = {"oneOf", "anyOf", "items", "properties", "additionalProperties"}
        return not any(key in item for key in container_keys)

    refs_in_nested_unions: set[str] = set()
    for item in items_any_of:
        if isinstance(item, dict) and not _is_direct_ref_item(item):
            _collect_refs(item, refs_in_nested_unions)

    if not refs_in_nested_unions:
        return openapi_schema

    # Remove sibling direct refs that are duplicates of refs found elsewhere
    deduplicated: list[Any] = []
    for item in items_any_of:
        if isinstance(item, dict) and _is_direct_ref_item(item) and item["$ref"] in refs_in_nested_unions:
            continue
        deduplicated.append(item)

    items_schema["anyOf"] = deduplicated
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

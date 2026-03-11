#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Responses API Integration Test Coverage Analyzer

The expected feature set is derived from the OpenAI API spec and the
llama-stack fastapi_routes.py files. Coverage detection uses AST analysis
of integration tests.

Usage:
    uv run python scripts/responses_test_coverage.py [--verbose]
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = ROOT / "tests" / "integration" / "responses"
OPENAI_SPEC = ROOT / "docs" / "static" / "openai-spec-2.3.0.yml"
AGENTS_ROUTES = ROOT / "src" / "llama_stack_api" / "agents" / "fastapi_routes.py"
CONVERSATIONS_ROUTES = ROOT / "src" / "llama_stack_api" / "conversations" / "fastapi_routes.py"


# ---------------------------------------------------------------------------
# Spec helpers
# ---------------------------------------------------------------------------


def _load_spec(path: Path) -> dict[str, Any]:
    content = path.read_text()
    if path.suffix in (".yml", ".yaml"):
        return yaml.safe_load(content)
    return json.loads(content)


def _resolve_ref(ref: str, spec: dict[str, Any]) -> dict[str, Any]:
    parts = ref.lstrip("#/").split("/")
    obj: Any = spec
    for p in parts:
        obj = obj[p]
    return obj


def _collect_properties(
    schema: dict[str, Any], spec: dict[str, Any], visited: set[str] | None = None
) -> dict[str, Any]:
    """Recursively collect all property names from a schema, resolving $ref."""
    if visited is None:
        visited = set()
    props: dict[str, Any] = {}
    if not isinstance(schema, dict):
        return props
    ref = schema.get("$ref")
    if ref:
        if ref in visited:
            return props
        visited.add(ref)
        return _collect_properties(_resolve_ref(ref, spec), spec, visited)
    if "properties" in schema:
        for name, val in schema["properties"].items():
            props[name] = val
    for key in ("allOf", "oneOf", "anyOf"):
        if key in schema:
            for sub in schema[key]:
                props.update(_collect_properties(sub, spec, visited))
    return props


def _get_type_value(schema: dict[str, Any], spec: dict[str, Any]) -> str | None:
    """Extract the 'type' discriminator value from a schema (e.g. 'function', 'response.created')."""
    # Direct properties
    tp = schema.get("properties", {}).get("type", {})
    if tp:
        return tp.get("enum", [None])[0] or tp.get("const")
    # Walk allOf
    for sub in schema.get("allOf", []):
        if "$ref" in sub:
            sub = _resolve_ref(sub["$ref"], spec)
        tp = sub.get("properties", {}).get("type", {})
        if tp:
            return tp.get("enum", [None])[0] or tp.get("const")
    return None


def _extract_oneof_types(ref: str, spec: dict[str, Any]) -> list[str]:
    """Extract all type discriminator values from a oneOf/anyOf union."""
    schema = _resolve_ref(ref, spec)
    types = []
    for key in ("oneOf", "anyOf"):
        for item in schema.get(key, []):
            if "$ref" in item:
                resolved = _resolve_ref(item["$ref"], spec)
                val = _get_type_value(resolved, spec)
                if val:
                    types.append(val)
    return types


# ---------------------------------------------------------------------------
# Route extraction from fastapi_routes.py
# ---------------------------------------------------------------------------

# Map route path prefixes to (category, sdk_resource)
_ROUTE_CATEGORY_MAP = {
    "/responses": ("CRUD Operations", "responses"),
    "/conversations": ("Conversations", "conversations"),
}


def _extract_routes_from_file(filepath: Path) -> list[tuple[str, str]]:
    """Parse a fastapi_routes.py and return (method, path) tuples."""
    source = filepath.read_text()
    tree = ast.parse(source)
    routes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "router":
                method = node.func.attr
                if method in ("get", "post", "put", "delete", "patch"):
                    if node.args and isinstance(node.args[0], ast.Constant):
                        path = node.args[0].value
                        routes.append((method, path))
    return routes


def _route_to_feature_id(method: str, path: str) -> str:
    """Convert a (method, path) to a feature id like 'crud.create' or 'conv.retrieve'."""
    # Determine prefix from path
    if path.startswith("/responses"):
        prefix = "crud"
    elif path.startswith("/conversations"):
        prefix = "conv"
    else:
        prefix = "api"

    # Generate suffix from method + path structure
    segments = [s for s in path.split("/") if s and not s.startswith("{")]
    # Remove the resource prefix (responses/conversations)
    sub_segments = segments[1:]  # e.g. ['input_items'] or []

    if method == "post" and not sub_segments:
        return f"{prefix}.create"
    if method == "get" and not sub_segments:
        return f"{prefix}.list"
    if method == "get" and "{" in path and not sub_segments:
        return f"{prefix}.retrieve"
    if method == "delete" and not sub_segments:
        return f"{prefix}.delete"

    # For paths with sub-resources like /responses/{id}/input_items
    if sub_segments:
        suffix = "_".join(sub_segments)
        if method == "get":
            return f"{prefix}.{suffix}"
        if method == "post":
            return f"{prefix}.{suffix}"
        if method == "delete":
            return f"{prefix}.delete_{suffix}"

    # Fallback for simple id-based routes
    if method == "get":
        return f"{prefix}.retrieve"
    if method == "delete":
        return f"{prefix}.delete"

    return f"{prefix}.{method}"


def _route_to_sdk_method(method: str, path: str) -> str:
    """Convert a (method, path) to the SDK method name like 'responses.create'."""
    if path.startswith("/responses"):
        base = "responses"
    elif path.startswith("/conversations"):
        base = "conversations"
    else:
        return f"unknown.{method}"

    segments = [s for s in path.split("/") if s and not s.startswith("{")]
    sub_segments = segments[1:]  # After 'responses' or 'conversations'

    if not sub_segments:
        method_map = {"post": "create", "get": "list", "delete": "delete"}
        # GET with {id} is retrieve, GET without is list
        if method == "get" and "{" in path:
            return f"{base}.retrieve"
        return f"{base}.{method_map.get(method, method)}"

    # Sub-resources: /responses/{id}/input_items -> responses.input_items.list
    sub = ".".join(sub_segments)
    if method == "get":
        return f"{base}.{sub}.list" if "{" not in sub_segments[-1] else f"{base}.{sub}"
    if method == "post":
        return f"{base}.{sub}"
    if method == "delete":
        return f"{base}.{sub}.delete"
    return f"{base}.{sub}.{method}"


def _route_to_description(method: str, path: str) -> str:
    return f"{method.upper()} {path}"


# Params to skip — always present or not meaningfully testable
_SKIP_PARAMS = {
    "prompt",  # internal/deprecated
    "user",  # identity param
    "stream_options",  # sub-option of stream
    "prompt_cache_retention",  # not yet supported
}


@dataclass
class Feature:
    """A testable feature of the Responses API."""

    id: str
    category: str
    description: str
    sdk_method: str = ""  # SDK method to match in tests (e.g. 'responses.create')
    property_names: list[str] = field(default_factory=list)
    covered: bool = False
    test_locations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Feature matrix builder — spec + routes driven
# ---------------------------------------------------------------------------


def build_feature_matrix(
    openai_spec_path: Path = OPENAI_SPEC,
    agents_routes_path: Path = AGENTS_ROUTES,
    conversations_routes_path: Path = CONVERSATIONS_ROUTES,
) -> list[Feature]:
    """Build feature list from the OpenAI API spec and fastapi route files."""
    spec = _load_spec(openai_spec_path)
    features: list[Feature] = []

    # --- Request parameters from POST /responses schema ---
    create_ref = spec["paths"]["/responses"]["post"]["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    req_props = _collect_properties({"$ref": create_ref}, spec)
    for name in sorted(req_props.keys()):
        if name in _SKIP_PARAMS:
            continue
        features.append(
            Feature(
                id=f"param.{name}",
                category="Request Parameters",
                description=f"{name} parameter",
                property_names=[name],
            )
        )

    # --- Tool types from Tool oneOf ---
    tool_types = _extract_oneof_types("#/components/schemas/Tool", spec)
    seen_types: set[str] = set()
    for type_val in tool_types:
        # Normalize versioned types like 'web_search_2025_08_26' -> 'web_search'
        normalized = type_val.split("_20")[0] if "_20" in type_val else type_val
        if normalized in seen_types:
            continue
        seen_types.add(normalized)
        features.append(
            Feature(
                id=f"tools.{normalized}",
                category="Tools",
                description=f"{normalized} tool",
                property_names=["tools"],
            )
        )
    # function_call_output (behavioral — multi-turn tool use)
    features.append(
        Feature(
            id="tools.function_call_output",
            category="Tools",
            description="function_call_output in multi-turn",
            property_names=["output"],
        )
    )

    # --- Structured output sub-features ---
    for fmt in ("json_schema", "json_object"):
        features.append(
            Feature(
                id=f"text.{fmt}",
                category="Structured Output",
                description=f"text format {fmt}",
                property_names=["text"],
            )
        )

    # --- Response validation (behavioral) ---
    for fid, desc, props in [
        ("resp.id_prefix", "response id starts with resp_", ["id"]),
        ("resp.status_completed", "status == completed", ["status"]),
        ("resp.output_text", "output_text content", ["output"]),
        ("resp.usage", "usage fields present", ["usage"]),
        ("resp.model_echo", "model echoed in response", ["model"]),
        ("resp.error", "error field on failure", ["error"]),
    ]:
        features.append(Feature(id=fid, category="Response Validation", description=desc, property_names=props))

    # --- Streaming events from ResponseStreamEvent oneOf ---
    event_types = _extract_oneof_types("#/components/schemas/ResponseStreamEvent", spec)
    for event_type in event_types:
        # event_type is like 'response.created', 'response.output_text.delta'
        short = event_type.replace("response.", "", 1)
        features.append(
            Feature(
                id=f"stream.{short}",
                category="Streaming Events",
                description=f"{event_type} event",
            )
        )

    # --- CRUD and Conversation endpoints from fastapi_routes.py ---
    for routes_path in (agents_routes_path, conversations_routes_path):
        if not routes_path.exists():
            continue
        for method, path in _extract_routes_from_file(routes_path):
            fid = _route_to_feature_id(method, path)
            sdk_method = _route_to_sdk_method(method, path)
            desc = _route_to_description(method, path)
            category = "CRUD Operations" if path.startswith("/responses") else "Conversations"
            features.append(Feature(id=fid, category=category, description=desc, sdk_method=sdk_method))

    # conversation= param in responses.create
    features.append(
        Feature(
            id="conv.with_response",
            category="Conversations",
            description="conversation= param in responses.create",
        )
    )

    # --- Error handling (behavioral) ---
    for fid, desc in [
        ("err.invalid_model", "invalid model raises error"),
        ("err.invalid_params", "invalid parameters raise error"),
        ("err.invalid_image", "invalid image input error"),
    ]:
        features.append(Feature(id=fid, category="Error Handling", description=desc))

    return features


# ---------------------------------------------------------------------------
# AST-based test analysis
# ---------------------------------------------------------------------------


def _get_call_chain(node: ast.Call) -> str | None:
    """Reconstruct dotted call chain like 'openai_client.responses.create'."""
    parts: list[str] = []
    obj = node.func
    while isinstance(obj, ast.Attribute):
        parts.append(obj.attr)
        obj = obj.value
    if isinstance(obj, ast.Name):
        parts.append(obj.id)
    else:
        return None
    parts.reverse()
    return ".".join(parts)


def _is_openai_call(chain: str) -> bool:
    return any(chain.startswith(c) for c in ("openai_client.", "alice_client.", "bob_client.", "self."))


def _strip_client_prefix(chain: str) -> str:
    """'openai_client.responses.create' -> 'responses.create'"""
    return chain.split(".", 1)[1] if "." in chain else chain


@dataclass
class TestEvidence:
    """Evidence of what a test exercises, extracted via AST."""

    params: set[str] = field(default_factory=set)
    tool_types: set[str] = field(default_factory=set)
    text_formats: set[str] = field(default_factory=set)
    api_methods: set[str] = field(default_factory=set)
    stream_events: set[str] = field(default_factory=set)
    error_types: set[str] = field(default_factory=set)
    response_attrs: set[str] = field(default_factory=set)
    has_function_call_output: bool = False


def _analyze_test_ast(func_node: ast.AST) -> TestEvidence:
    """Walk a test function's AST and extract coverage evidence."""
    ev = TestEvidence()

    for node in ast.walk(func_node):
        # --- API calls and their keyword args ---
        if isinstance(node, ast.Call):
            chain = _get_call_chain(node)
            if chain and _is_openai_call(chain):
                method = _strip_client_prefix(chain)
                ev.api_methods.add(method)
                if method == "responses.create":
                    for kw in node.keywords:
                        if kw.arg:
                            ev.params.add(kw.arg)

        # --- Dict literals: detect tool types and text formats ---
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values, strict=False):
                if isinstance(k, ast.Constant) and isinstance(v, ast.Constant) and isinstance(v.value, str):
                    if k.value == "type":
                        # Normalize versioned types
                        val = v.value.split("_20")[0] if "_20" in v.value else v.value
                        ev.tool_types.add(val)
                        if val in ("json_schema", "json_object"):
                            ev.text_formats.add(val)
                        if val == "function_call_output":
                            ev.has_function_call_output = True

        # --- String constants: detect stream event types ---
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            val = node.value
            if val.startswith("response.") and len(val) > len("response."):
                ev.stream_events.add(val)
            if val == "resp_":
                ev.response_attrs.add("id_prefix")
            if val in ("completed", "queued", "in_progress", "failed"):
                ev.response_attrs.add("status")
            if val in ("invalid_base64_image", "server_error", "invalid_base64"):
                ev.error_types.add("invalid_image")
            if val in ("validation", "invalid", "bad_request"):
                ev.error_types.add("invalid_params")

        # --- Attribute access on response objects ---
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in ("response", "response1", "response2", "retrieved"):
                ev.response_attrs.add(node.attr)

        # --- Exception types ---
        if isinstance(node, ast.Name):
            if node.id == "NotFoundError":
                ev.error_types.add("invalid_model")
            elif node.id == "BadRequestError":
                ev.error_types.add("invalid_params")

    return ev


# ---------------------------------------------------------------------------
# Test scanning and matching
# ---------------------------------------------------------------------------


def _extract_openai_test_functions(filepath: Path) -> list[tuple[str, ast.AST]]:
    """Return (location_key, func_node) for tests that use openai_client."""
    source = filepath.read_text()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        print(f"  WARNING: Could not parse {filepath}", file=sys.stderr)
        return []

    results = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith("test_"):
            arg_names = [arg.arg for arg in node.args.args]
            if any(a in arg_names for a in ("openai_client", "alice_client", "bob_client")):
                location = f"{filepath.relative_to(ROOT)}:{node.lineno}::{node.name}"
                results.append((location, node))
    return results


def _scan_streaming_helpers(test_dir: Path) -> set[str]:
    """Extract stream event types from streaming_assertions.py."""
    helpers = test_dir / "streaming_assertions.py"
    if not helpers.exists():
        return set()
    source = helpers.read_text()
    tree = ast.parse(source)
    events: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            val = node.value
            if val.startswith("response.") and len(val) > len("response."):
                events.add(val)
    return events


def _match_evidence(features: list[Feature], evidence_map: dict[str, TestEvidence], helper_events: set[str]) -> None:
    """Match accumulated test evidence against features."""
    # Aggregate evidence
    all_params: dict[str, list[str]] = {}
    all_tool_types: dict[str, list[str]] = {}
    all_text_formats: dict[str, list[str]] = {}
    all_api_methods: dict[str, list[str]] = {}
    all_stream_events: dict[str, list[str]] = {}
    all_error_types: dict[str, list[str]] = {}
    all_response_attrs: dict[str, list[str]] = {}
    has_function_call_output: list[str] = []

    for loc, ev in evidence_map.items():
        for p in ev.params:
            all_params.setdefault(p, []).append(loc)
        for t in ev.tool_types:
            all_tool_types.setdefault(t, []).append(loc)
        for f in ev.text_formats:
            all_text_formats.setdefault(f, []).append(loc)
        for m in ev.api_methods:
            all_api_methods.setdefault(m, []).append(loc)
        for e in ev.stream_events:
            all_stream_events.setdefault(e, []).append(loc)
        for e in ev.error_types:
            all_error_types.setdefault(e, []).append(loc)
        for a in ev.response_attrs:
            all_response_attrs.setdefault(a, []).append(loc)
        if ev.has_function_call_output:
            has_function_call_output.append(loc)

    for e in helper_events:
        all_stream_events.setdefault(e, []).append("streaming_assertions.py")

    for feat in features:
        locs: list[str] = []

        if feat.id.startswith("param."):
            param_name = feat.id[len("param.") :]
            locs = all_params.get(param_name, [])

        elif feat.id.startswith("tools.") and feat.id != "tools.function_call_output":
            tool_type = feat.id[len("tools.") :]
            locs = all_tool_types.get(tool_type, [])

        elif feat.id == "tools.function_call_output":
            locs = has_function_call_output

        elif feat.id.startswith("text."):
            fmt = feat.id[len("text.") :]
            locs = all_text_formats.get(fmt, [])

        elif feat.id.startswith("resp."):
            suffix = feat.id[len("resp.") :]
            attr_map = {
                "id_prefix": "id_prefix",
                "status_completed": "status",
                "output_text": "output_text",
                "usage": "usage",
                "model_echo": "model",
                "error": "error",
            }
            attr = attr_map.get(suffix)
            if attr:
                locs = all_response_attrs.get(attr, [])

        elif feat.id.startswith("stream."):
            event_type = "response." + feat.id[len("stream.") :]
            locs = all_stream_events.get(event_type, [])

        elif feat.id == "conv.with_response":
            locs = all_params.get("conversation", [])

        elif feat.id.startswith("err."):
            error_type = feat.id[len("err.") :]
            locs = all_error_types.get(error_type, [])

        elif feat.sdk_method:
            # CRUD / Conversations — match by SDK method
            locs = all_api_methods.get(feat.sdk_method, [])

        if locs:
            feat.covered = True
            feat.test_locations = list(dict.fromkeys(locs))


def run_coverage(test_dir: Path = TESTS_DIR, spec_path: Path = OPENAI_SPEC) -> list[Feature]:
    """Build features from spec, scan tests via AST, and match coverage."""
    features = build_feature_matrix(spec_path)

    evidence_map: dict[str, TestEvidence] = {}
    for filepath in sorted(test_dir.glob("test_*.py")):
        for location, func_node in _extract_openai_test_functions(filepath):
            evidence_map[location] = _analyze_test_ast(func_node)

    helper_events = _scan_streaming_helpers(test_dir)
    _match_evidence(features, evidence_map, helper_events)

    return features


def get_tested_property_names(features: list[Feature] | None = None) -> set[str]:
    """Return the set of OpenAI spec property names that have integration test coverage."""
    if features is None:
        features = run_coverage()

    tested = set()
    for feat in features:
        if feat.covered:
            tested.update(feat.property_names)
    return tested


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

CATEGORY_ORDER = [
    "Request Parameters",
    "Tools",
    "Structured Output",
    "Response Validation",
    "Streaming Events",
    "CRUD Operations",
    "Conversations",
    "Error Handling",
]


def print_report(features: list[Feature], verbose: bool = False) -> float:
    """Print coverage report and return overall score."""
    categories: dict[str, list[Feature]] = {}
    for feat in features:
        categories.setdefault(feat.category, []).append(feat)

    total = len(features)
    covered = sum(1 for feat in features if feat.covered)
    score = (covered / total * 100) if total > 0 else 0

    print("=" * 72)
    print("  Responses API — OpenAI Client Integration Test Coverage")
    print("=" * 72)
    print()
    print(f"  Overall Score: {score:.1f}% ({covered}/{total} features covered)")
    print()

    print(f"{'Category':<25} {'Covered':>8} {'Total':>8} {'Score':>8}")
    print("-" * 55)
    for cat_name in CATEGORY_ORDER:
        cat_features = categories.get(cat_name, [])
        if not cat_features:
            continue
        cat_covered = sum(1 for feat in cat_features if feat.covered)
        cat_total = len(cat_features)
        cat_score = (cat_covered / cat_total * 100) if cat_total > 0 else 0
        print(f"{cat_name:<25} {cat_covered:>8} {cat_total:>8} {cat_score:>7.1f}%")

    print()

    gaps = [feat for feat in features if not feat.covered]
    if gaps:
        print(f"GAPS ({len(gaps)} features missing coverage):")
        print()
        current_cat = None
        for feat in gaps:
            if feat.category != current_cat:
                current_cat = feat.category
                print(f"  [{current_cat}]")
            print(f"    - {feat.id}: {feat.description}")
        print()

    if verbose:
        print("COVERED FEATURES:")
        print()
        current_cat = None
        for feat in features:
            if not feat.covered:
                continue
            if feat.category != current_cat:
                current_cat = feat.category
                print(f"  [{current_cat}]")
            locs = ", ".join(feat.test_locations[:3])
            if len(feat.test_locations) > 3:
                locs += f" (+{len(feat.test_locations) - 3} more)"
            print(f"    + {feat.id}: {feat.description}")
            print(f"      {locs}")
        print()

    return score


def main() -> None:
    parser = argparse.ArgumentParser(description="Responses API test coverage analyzer")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show covered features with test locations")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if not TESTS_DIR.exists():
        print(f"ERROR: Test directory not found: {TESTS_DIR}", file=sys.stderr)
        sys.exit(1)

    features = run_coverage()

    if args.json:
        data = {
            "score": round(sum(1 for f in features if f.covered) / len(features) * 100, 1),
            "total": len(features),
            "covered": sum(1 for f in features if f.covered),
            "tested_properties": sorted(get_tested_property_names(features)),
            "gaps": [
                {"id": f.id, "category": f.category, "description": f.description} for f in features if not f.covered
            ],
            "covered_features": [
                {
                    "id": f.id,
                    "category": f.category,
                    "description": f.description,
                    "test_locations": f.test_locations,
                }
                for f in features
                if f.covered
            ],
        }
        print(json.dumps(data, indent=2))
    else:
        print(f"\nScanned tests from {TESTS_DIR.relative_to(ROOT)}/\n")
        print_report(features, verbose=args.verbose)


if __name__ == "__main__":
    main()

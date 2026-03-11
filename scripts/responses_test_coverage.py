#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Responses API Integration Test Coverage Analyzer

Scans integration tests under tests/integration/responses/ to determine
which OpenAI Responses API parameters and features are exercised via the
OpenAI client (not the llama-stack client).

Features are derived from the OpenAI API spec file, not hardcoded.

Produces a coverage score and a gap report.

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
LLAMA_SPEC = ROOT / "docs" / "static" / "llama-stack-spec.yaml"


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


def _collect_oneof_names(ref: str, spec: dict[str, Any]) -> list[str]:
    """Return schema names from a oneOf/anyOf union."""
    schema = _resolve_ref(ref, spec)
    names = []
    for key in ("oneOf", "anyOf"):
        for item in schema.get(key, []):
            if "$ref" in item:
                names.append(item["$ref"].split("/")[-1])
    return names


# ---------------------------------------------------------------------------
# Keyword generators — map spec property names to test-detection keywords
# ---------------------------------------------------------------------------

# Some params need special keyword patterns (e.g. "input" is too generic)
_PARAM_KEYWORD_OVERRIDES: dict[str, list[str]] = {
    "input": ['input="', "input='", "input=[{", 'input=[{"role"'],
    "stream": ["stream=True"],
    "text": ["json_schema", "json_object", '"format"'],
    "tools": ['"type": "function"', "'type': 'function'", "web_search", "file_search", '"type": "mcp"'],
    "model": ["model="],
}

# Params to skip — always present or not meaningfully testable in isolation
_SKIP_PARAMS = {
    "prompt",  # internal/deprecated
    "user",  # identity param, not testable
    "stream_options",  # sub-option of stream
    "prompt_cache_retention",  # not yet supported
}


def _keywords_for_param(name: str) -> list[str]:
    """Return keyword patterns to search for in test sources for a given param."""
    if name in _PARAM_KEYWORD_OVERRIDES:
        return _PARAM_KEYWORD_OVERRIDES[name]
    return [f"{name}="]


# Tool type schema name → (feature id suffix, description, keywords)
_TOOL_TYPE_MAP: dict[str, tuple[str, str, list[str]]] = {
    "FunctionTool": ("function_def", "function tool definition", ['"type": "function"', "'type': 'function'"]),
    "WebSearchTool": ("web_search", "web_search tool", ["web_search", "web-search"]),
    "WebSearchPreviewTool": ("web_search", "web_search tool", ["web_search", "web-search"]),
    "FileSearchTool": ("file_search", "file_search tool", ["file_search", "file-search"]),
    "MCPTool": ("mcp", "MCP tool", ['"type": "mcp"', "'type': 'mcp'"]),
    "CodeInterpreterTool": ("code_interpreter", "code_interpreter tool", ["code_interpreter"]),
    "ComputerUsePreviewTool": ("computer_use", "computer_use tool", ["computer_use", "computer-use"]),
    "ImageGenTool": ("image_gen", "image generation tool", ["image_gen", "image-gen"]),
}

# Streaming event schema name → (event dotted name, keywords)
_STREAM_EVENT_MAP: dict[str, tuple[str, list[str]]] = {
    "ResponseCreatedEvent": ("response.created", ["response.created"]),
    "ResponseCompletedEvent": ("response.completed", ["response.completed"]),
    "ResponseFailedEvent": ("response.failed", ["response.failed"]),
    "ResponseInProgressEvent": ("response.in_progress", ["response.in_progress", "in_progress"]),
    "ResponseIncompleteEvent": ("response.incomplete", ["response.incomplete", "incomplete"]),
    "ResponseOutputItemAddedEvent": ("response.output_item.added", ["output_item.added", "output_item_added"]),
    "ResponseOutputItemDoneEvent": ("response.output_item.done", ["output_item.done", "output_item_done"]),
    "ResponseContentPartAddedEvent": ("response.content_part.added", ["content_part.added", "content_part_added"]),
    "ResponseContentPartDoneEvent": ("response.content_part.done", ["content_part.done", "content_part_done"]),
    "ResponseTextDeltaEvent": ("response.output_text.delta", ["output_text.delta", "output_text_delta", "TextDelta"]),
    "ResponseTextDoneEvent": ("response.output_text.done", ["output_text.done", "output_text_done", "TextDone"]),
    "ResponseFunctionCallArgumentsDeltaEvent": (
        "response.function_call_arguments.delta",
        ["function_call_arguments.delta"],
    ),
    "ResponseFunctionCallArgumentsDoneEvent": (
        "response.function_call_arguments.done",
        ["function_call_arguments.done"],
    ),
    "ResponseQueuedEvent": ("response.queued", ["response.queued", "queued"]),
    "ResponseErrorEvent": ("response.error", ["response.error"]),
}

# CRUD endpoint → (feature id, description, keywords)
_CRUD_ENDPOINTS: dict[str, dict[str, tuple[str, str, list[str]]]] = {
    "/responses": {
        "post": ("crud.create", "POST /responses (create)", ["responses.create"]),
        "get": ("crud.list", "GET /responses (list)", ["responses.list"]),
    },
    "/responses/{response_id}": {
        "get": ("crud.retrieve", "GET /responses/{id} (retrieve)", ["responses.retrieve"]),
        "delete": ("crud.delete", "DELETE /responses/{id} (delete)", ["responses.delete", "responses.del"]),
    },
    "/responses/{response_id}/input_items": {
        "get": ("crud.input_items", "GET /responses/{id}/input_items", ["input_items", "list_input_items"]),
    },
    "/responses/{response_id}/cancel": {
        "post": ("crud.cancel", "POST /responses/{id}/cancel", ["responses.cancel"]),
    },
}

_CONVERSATION_ENDPOINTS: dict[str, dict[str, tuple[str, str, list[str]]]] = {
    "/conversations": {
        "post": ("conv.create", "create conversation", ["conversations.create"]),
    },
    "/conversations/{conversation_id}": {
        "get": ("conv.retrieve", "retrieve conversation", ["conversations.retrieve"]),
        "delete": ("conv.delete", "delete conversation", ["conversations.delete", "conversations.del"]),
    },
    "/conversations/{conversation_id}/items": {
        "get": ("conv.list_items", "list conversation items", ["conversations.items", "conversations.list"]),
    },
}


@dataclass
class Feature:
    """A testable feature of the Responses API."""

    id: str
    category: str
    description: str
    keywords: list[str]
    property_names: list[str] = field(default_factory=list)
    covered: bool = False
    test_locations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Feature matrix builder — spec-driven
# ---------------------------------------------------------------------------


def build_feature_matrix(openai_spec_path: Path = OPENAI_SPEC) -> list[Feature]:
    """Build feature list from the OpenAI API spec."""
    spec = _load_spec(openai_spec_path)
    features: list[Feature] = []

    # --- Request parameters ---
    create_ref = spec["paths"]["/responses"]["post"]["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    req_props = _collect_properties({"$ref": create_ref}, spec)

    for name in sorted(req_props.keys()):
        if name in _SKIP_PARAMS:
            continue
        keywords = _keywords_for_param(name)
        features.append(
            Feature(
                id=f"param.{name}",
                category="Request Parameters",
                description=f"{name} parameter",
                keywords=keywords,
                property_names=[name],
            )
        )

    # --- Tool types ---
    tool_names = _collect_oneof_names("#/components/schemas/Tool", spec)
    seen_tool_ids: set[str] = set()
    for schema_name in tool_names:
        if schema_name not in _TOOL_TYPE_MAP:
            continue
        suffix, desc, kws = _TOOL_TYPE_MAP[schema_name]
        fid = f"tools.{suffix}"
        if fid in seen_tool_ids:
            continue
        seen_tool_ids.add(fid)
        features.append(Feature(id=fid, category="Tools", description=desc, keywords=kws, property_names=["tools"]))

    # function_call_output (multi-turn tool use) — behavioral feature
    features.append(
        Feature(
            id="tools.function_call_output",
            category="Tools",
            description="function_call_output in multi-turn",
            keywords=["function_call_output", "call_id"],
            property_names=["output"],
        )
    )

    # --- Structured output (sub-features of the "text" param) ---
    features.append(
        Feature(
            id="text.json_schema",
            category="Structured Output",
            description="text format json_schema",
            keywords=["json_schema"],
            property_names=["text"],
        )
    )
    features.append(
        Feature(
            id="text.json_object",
            category="Structured Output",
            description="text format json_object",
            keywords=["json_object"],
            property_names=["text"],
        )
    )

    # --- Response validation (behavioral) ---
    for fid, desc, kws, props in [
        ("resp.id_prefix", "response id starts with resp_", ['resp_"', "resp_'", 'startswith("resp_'], ["id"]),
        ("resp.status_completed", "status == completed", [".status", "completed"], ["status"]),
        ("resp.output_text", "output_text content", ["output_text"], ["output"]),
        ("resp.usage", "usage fields present", [".usage", "input_tokens", "output_tokens"], ["usage"]),
        ("resp.model_echo", "model echoed in response", ["response.model", ".model ==", ".model="], ["model"]),
        ("resp.error", "error field on failure", [".error", "response.error"], ["error"]),
    ]:
        features.append(
            Feature(id=fid, category="Response Validation", description=desc, keywords=kws, property_names=props)
        )

    # --- Streaming events ---
    event_names = _collect_oneof_names("#/components/schemas/ResponseStreamEvent", spec)
    for schema_name in event_names:
        if schema_name not in _STREAM_EVENT_MAP:
            continue
        dotted, kws = _STREAM_EVENT_MAP[schema_name]
        features.append(
            Feature(
                id=f"stream.{dotted.replace('response.', '')}",
                category="Streaming Events",
                description=f"{dotted} event",
                keywords=kws,
            )
        )

    # --- CRUD operations ---
    for path, methods in _CRUD_ENDPOINTS.items():
        if path.replace("{response_id}", "{id}").lstrip("/") not in str(spec.get("paths", {})):
            pass  # still add — we track desired coverage not just implemented
        for _method, (fid, desc, kws) in methods.items():
            features.append(Feature(id=fid, category="CRUD Operations", description=desc, keywords=kws))

    # --- Conversations ---
    for _path, methods in _CONVERSATION_ENDPOINTS.items():
        for _method, (fid, desc, kws) in methods.items():
            features.append(Feature(id=fid, category="Conversations", description=desc, keywords=kws))
    features.append(
        Feature(
            id="conv.with_response",
            category="Conversations",
            description="conversation= param in responses.create",
            keywords=["conversation="],
        )
    )

    # --- Error handling (behavioral) ---
    features.append(
        Feature(
            id="err.invalid_model",
            category="Error Handling",
            description="invalid model raises error",
            keywords=["invalid", "not_found", "NotFoundError"],
        )
    )
    features.append(
        Feature(
            id="err.invalid_params",
            category="Error Handling",
            description="invalid parameters raise error",
            keywords=["BadRequestError", "bad_request", "validation"],
        )
    )
    features.append(
        Feature(
            id="err.invalid_image",
            category="Error Handling",
            description="invalid image input error",
            keywords=["invalid_base64", "invalid_image", "image_parse_error"],
        )
    )

    return features


# ---------------------------------------------------------------------------
# Test scanning
# ---------------------------------------------------------------------------


def extract_test_functions(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Extract all test functions/methods from an AST."""
    tests = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if node.name.startswith("test_"):
                tests.append(node)
    return tests


def get_openai_client_tests(filepath: Path, source: str, tree: ast.Module) -> list[tuple[str, str]]:
    """Return (test_name, test_source) pairs for tests using openai_client."""
    results = []
    lines = source.splitlines()

    for func in extract_test_functions(tree):
        arg_names = [arg.arg for arg in func.args.args]

        # openai_client or access-control clients (which are also openai clients)
        uses_openai = any(a in arg_names for a in ("openai_client", "alice_client", "bob_client"))

        if not uses_openai:
            continue

        start = func.lineno - 1
        end = func.end_lineno if func.end_lineno else start + 1
        func_source = "\n".join(lines[start:end])
        location = f"{filepath.relative_to(ROOT)}:{func.lineno}"
        results.append((f"{location}::{func.name}", func_source))

    return results


def scan_tests(test_dir: Path) -> list[tuple[str, str]]:
    """Scan all test files and return openai_client test sources."""
    all_tests = []
    for filepath in sorted(test_dir.glob("test_*.py")):
        source = filepath.read_text()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            print(f"  WARNING: Could not parse {filepath}", file=sys.stderr)
            continue

        all_tests.extend(get_openai_client_tests(filepath, source, tree))

    return all_tests


def match_coverage(features: list[Feature], tests: list[tuple[str, str]]) -> None:
    """Match test sources against feature keywords."""
    for feat in features:
        for test_name, test_source in tests:
            for kw in feat.keywords:
                if kw in test_source:
                    feat.covered = True
                    if test_name not in feat.test_locations:
                        feat.test_locations.append(test_name)
                    break


def scan_streaming_helpers(test_dir: Path, features: list[Feature]) -> None:
    """Check if streaming helper validates specific events (used by openai_client tests)."""
    helpers = test_dir / "streaming_assertions.py"
    if not helpers.exists():
        return
    source = helpers.read_text()
    for feat in features:
        if feat.category == "Streaming Events":
            for kw in feat.keywords:
                if kw in source:
                    feat.covered = True
                    loc = "tests/integration/responses/streaming_assertions.py"
                    if loc not in feat.test_locations:
                        feat.test_locations.append(loc)


def get_tested_property_names(features: list[Feature] | None = None) -> set[str]:
    """Return the set of OpenAI spec property names that have integration test coverage.

    This is the main entry point for cross-referencing with the conformance report.
    """
    if features is None:
        features = build_feature_matrix()
        if TESTS_DIR.exists():
            tests = scan_tests(TESTS_DIR)
            match_coverage(features, tests)
            scan_streaming_helpers(TESTS_DIR, features)

    tested = set()
    for feat in features:
        if feat.covered:
            tested.update(feat.property_names)
    return tested


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

    # Gaps
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

    features = build_feature_matrix()
    tests = scan_tests(TESTS_DIR)
    match_coverage(features, tests)
    scan_streaming_helpers(TESTS_DIR, features)

    if args.json:
        data = {
            "score": round(sum(1 for feat in features if feat.covered) / len(features) * 100, 1),
            "total": len(features),
            "covered": sum(1 for feat in features if feat.covered),
            "tested_properties": sorted(get_tested_property_names(features)),
            "gaps": [
                {"id": feat.id, "category": feat.category, "description": feat.description}
                for feat in features
                if not feat.covered
            ],
            "covered_features": [
                {
                    "id": feat.id,
                    "category": feat.category,
                    "description": feat.description,
                    "test_locations": feat.test_locations,
                }
                for feat in features
                if feat.covered
            ],
        }
        print(json.dumps(data, indent=2))
    else:
        print(f"\nScanned {len(tests)} openai_client tests from {TESTS_DIR.relative_to(ROOT)}/\n")
        print_report(features, verbose=args.verbose)


if __name__ == "__main__":
    main()

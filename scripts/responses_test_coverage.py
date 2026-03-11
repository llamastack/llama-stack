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

ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = ROOT / "tests" / "integration" / "responses"


@dataclass
class Feature:
    """A testable feature of the Responses API."""

    id: str
    category: str
    description: str
    keywords: list[str]
    # Property names from the OpenAI spec that this feature covers
    # (used for cross-referencing with conformance report)
    property_names: list[str] = field(default_factory=list)
    covered: bool = False
    test_locations: list[str] = field(default_factory=list)


def build_feature_matrix() -> list[Feature]:
    """Build the full list of features we expect to be tested via openai_client."""
    features: list[Feature] = []

    def f(id: str, cat: str, desc: str, kw: list[str], props: list[str] | None = None) -> None:
        features.append(Feature(id=id, category=cat, description=desc, keywords=kw, property_names=props or []))

    # -- Request parameters (POST /responses) --
    cat = "Request Parameters"
    f("param.input_string", cat, "input as plain string", ['input="', "input='"], ["input"])
    f("param.input_messages", cat, "input as message list", ["input=[{", 'input=[{"role"'], ["input"])
    f("param.model", cat, "model parameter", ["model="], ["model"])
    f("param.instructions", cat, "instructions (system prompt)", ["instructions="], ["instructions"])
    f("param.temperature", cat, "temperature parameter", ["temperature="], ["temperature"])
    f("param.top_p", cat, "top_p parameter", ["top_p="], ["top_p"])
    f("param.frequency_penalty", cat, "frequency_penalty parameter", ["frequency_penalty="], ["frequency_penalty"])
    f("param.presence_penalty", cat, "presence_penalty parameter", ["presence_penalty="], ["presence_penalty"])
    f("param.max_output_tokens", cat, "max_output_tokens parameter", ["max_output_tokens="], ["max_output_tokens"])
    f("param.stream", cat, "stream=True", ["stream=True"], ["stream"])
    f("param.store", cat, "store parameter", ["store="], ["store"])
    f("param.metadata", cat, "metadata parameter", ["metadata="], ["metadata"])
    f("param.truncation", cat, "truncation parameter", ["truncation="], ["truncation"])
    f(
        "param.parallel_tool_calls",
        cat,
        "parallel_tool_calls parameter",
        ["parallel_tool_calls="],
        ["parallel_tool_calls"],
    )
    f("param.tool_choice", cat, "tool_choice parameter", ["tool_choice="], ["tool_choice"])
    f("param.max_tool_calls", cat, "max_tool_calls parameter", ["max_tool_calls="], ["max_tool_calls"])
    f("param.service_tier", cat, "service_tier parameter", ["service_tier="], ["service_tier"])
    f("param.top_logprobs", cat, "top_logprobs parameter", ["top_logprobs="], ["top_logprobs"])
    f("param.reasoning", cat, "reasoning parameter", ["reasoning="], ["reasoning"])
    f(
        "param.previous_response_id",
        cat,
        "previous_response_id for multi-turn",
        ["previous_response_id="],
        ["previous_response_id"],
    )
    f("param.background", cat, "background parameter", ["background="], ["background"])
    f("param.prompt_cache_key", cat, "prompt_cache_key parameter", ["prompt_cache_key="], ["prompt_cache_key"])
    f("param.safety_identifier", cat, "safety_identifier parameter", ["safety_identifier="], ["safety_identifier"])
    f("param.include", cat, "include parameter", ["include="], ["include"])

    # -- Tools --
    cat = "Tools"
    f("tools.function_def", cat, "function tool definition", ['"type": "function"', "'type': 'function'"], ["tools"])
    f("tools.web_search", cat, "web_search tool", ["web_search", "web-search"], ["tools"])
    f("tools.file_search", cat, "file_search tool", ["file_search", "file-search"], ["tools"])
    f("tools.mcp", cat, "MCP tool", ['"type": "mcp"', "'type': 'mcp'"], ["tools"])
    f(
        "tools.function_call_output",
        cat,
        "function_call_output in multi-turn",
        ["function_call_output", "call_id"],
        ["output"],
    )

    # -- Structured Output --
    cat = "Structured Output"
    f("text.json_schema", cat, "text format json_schema", ["json_schema"], ["text"])
    f("text.json_object", cat, "text format json_object", ["json_object"], ["text"])

    # -- Response validation --
    cat = "Response Validation"
    f("resp.id_prefix", cat, "response id starts with resp_", ['resp_"', "resp_'", 'startswith("resp_'], ["id"])
    f("resp.status_completed", cat, "status == completed", [".status", "completed"], ["status"])
    f("resp.output_text", cat, "output_text content", ["output_text"], ["output"])
    f("resp.usage", cat, "usage fields present", [".usage", "input_tokens", "output_tokens"], ["usage"])
    f("resp.model_echo", cat, "model echoed in response", ["response.model", ".model ==", ".model="], ["model"])
    f("resp.error", cat, "error field on failure", [".error", "response.error"], ["error"])

    # -- Streaming events --
    cat = "Streaming Events"
    f("stream.created", cat, "response.created event", ["response.created"])
    f("stream.completed", cat, "response.completed event", ["response.completed"])
    f("stream.failed", cat, "response.failed event", ["response.failed"])
    f(
        "stream.output_item_added",
        cat,
        "response.output_item.added event",
        ["output_item.added", "output_item_added"],
    )
    f(
        "stream.output_item_done",
        cat,
        "response.output_item.done event",
        ["output_item.done", "output_item_done"],
    )
    f(
        "stream.content_part_added",
        cat,
        "response.content_part.added event",
        ["content_part.added", "content_part_added"],
    )
    f(
        "stream.content_part_done",
        cat,
        "response.content_part.done event",
        ["content_part.done", "content_part_done"],
    )
    f(
        "stream.output_text_delta",
        cat,
        "response.output_text.delta event",
        ["output_text.delta", "output_text_delta"],
    )

    # -- CRUD operations --
    cat = "CRUD Operations"
    f("crud.create", cat, "POST /responses (create)", ["responses.create"])
    f("crud.retrieve", cat, "GET /responses/{id} (retrieve)", ["responses.retrieve"])
    f("crud.list", cat, "GET /responses (list)", ["responses.list"])
    f("crud.delete", cat, "DELETE /responses/{id} (delete)", ["responses.delete", "responses.del"])
    f("crud.input_items", cat, "GET /responses/{id}/input_items", ["input_items", "list_input_items"])

    # -- Conversations --
    cat = "Conversations"
    f("conv.create", cat, "create conversation", ["conversations.create"])
    f("conv.retrieve", cat, "retrieve conversation", ["conversations.retrieve"])
    f("conv.list_items", cat, "list conversation items", ["conversation_id", "conversations.list"])
    f("conv.with_response", cat, "conversation= param in responses.create", ["conversation="])

    # -- Error handling --
    cat = "Error Handling"
    f("err.invalid_model", cat, "invalid model raises error", ["invalid", "not_found", "NotFoundError"])
    f(
        "err.invalid_params",
        cat,
        "invalid parameters raise error",
        ["BadRequestError", "bad_request", "validation"],
    )
    f(
        "err.invalid_image",
        cat,
        "invalid image input error",
        ["invalid_base64", "invalid_image", "image_parse_error"],
    )

    return features


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

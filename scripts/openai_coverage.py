#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
OpenAI API Coverage Analyzer

Uses oasdiff to compare Llama Stack's OpenAPI spec against OpenAI's spec
and generates a coverage report showing:
- Which endpoints are implemented
- Which properties are conformant vs have issues
- A conformance score that increases as issues are fixed

Usage:
    python scripts/openai_coverage.py [--update]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


def _load_endpoint_categories(openai_spec: Path) -> dict[str, list[str]]:
    """Extract endpoint categories from OpenAI spec tags.

    Returns a mapping of category name to list of path prefixes.
    """
    spec = yaml.safe_load(openai_spec.read_text())
    categories: dict[str, list[str]] = defaultdict(list)

    for path, path_item in (spec.get("paths") or {}).items():
        if not isinstance(path_item, dict):
            continue
        for _, operation in path_item.items():
            if not isinstance(operation, dict):
                continue
            tags = operation.get("tags", [])
            if tags:
                # Use first tag as category
                category = tags[0]
                if path not in categories[category]:
                    categories[category].append(path)

    return dict(categories)


def _categorize_path(path: str, categories: dict[str, list[str]]) -> str:
    """Categorize an endpoint path using the loaded categories."""
    for category, paths in categories.items():
        for cat_path in paths:
            # Match if the path starts with the category path (handles path params)
            if path == cat_path or path.startswith(cat_path.rstrip("/") + "/") or path.startswith(cat_path + "{"):
                return category
    return "Other"


def _check_oasdiff_installed() -> bool:
    """Check if oasdiff is installed and accessible."""
    return shutil.which("oasdiff") is not None


def _run_oasdiff(openai_spec: Path, llama_spec: Path) -> dict[str, Any]:
    """Run oasdiff and return the JSON diff."""
    if not _check_oasdiff_installed():
        print("❌ Error: oasdiff is not installed or not in PATH")
        print()
        print("See installation instructions at:")
        print("  https://github.com/oasdiff/oasdiff#installation")
        print()
        sys.exit(1)

    result = subprocess.run(
        [
            "oasdiff",
            "diff",
            str(openai_spec),
            str(llama_spec),
            "--format",
            "json",
            "--strip-prefix-revision",
            "/v1",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0 and not result.stdout:
        raise RuntimeError(f"oasdiff failed: {result.stderr}")

    return json.loads(result.stdout) if result.stdout else {}


def _extract_issues(obj: Any, path: str = "") -> dict[str, Any]:
    """Recursively extract issues from oasdiff output."""
    result: dict[str, Any] = {
        "missing": [],  # Properties in OpenAI but not in Llama Stack
        "issues": [],  # Properties that exist but have conformance issues
    }

    if not isinstance(obj, dict):
        return result

    # Track deleted properties (missing from Llama Stack)
    if "deleted" in obj:
        deleted = obj["deleted"]
        if isinstance(deleted, list):
            for item in deleted:
                if isinstance(item, str):
                    prop_path = f"{path}.{item}" if path else item
                    prop_path = prop_path.replace(".modified.", ".")
                    result["missing"].append(prop_path)
        elif isinstance(deleted, dict):
            for location, items in deleted.items():
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, str):
                            prop_path = f"{path}.{location}.{item}" if path else f"{location}.{item}"
                            prop_path = prop_path.replace(".modified.", ".")
                            result["missing"].append(prop_path)

    # Track modified properties with issues
    if "modified" in obj and isinstance(obj["modified"], dict):
        for prop_name, prop_diff in obj["modified"].items():
            if not isinstance(prop_diff, dict):
                continue

            # Check if this is a container (no schema indicators)
            schema_indicators = {"enum", "type", "listOfTypes", "anyOf", "oneOf", "default", "schema"}
            if not any(key in prop_diff for key in schema_indicators):
                nested_path = f"{path}.{prop_name}" if path else prop_name
                nested = _extract_issues({"modified": prop_diff}, nested_path)
                result["missing"].extend(nested["missing"])
                result["issues"].extend(nested["issues"])
                continue

            clean_name = prop_name.replace(".modified.", ".")
            prop_path = f"{path}.{clean_name}" if path else clean_name
            prop_path = prop_path.replace(".modified.", ".")

            issue_details = []

            # Check for enum issues
            if "enum" in prop_diff:
                enum_diff = prop_diff["enum"]
                if enum_diff.get("enumDeleted"):
                    issue_details.append(f"Enum removed: {enum_diff.get('deleted', [])}")
                elif "deleted" in enum_diff:
                    issue_details.append(f"Enum values removed: {enum_diff['deleted']}")

            # Check for type issues
            if "type" in prop_diff:
                type_diff = prop_diff["type"]
                if "deleted" in type_diff:
                    issue_details.append(f"Type removed: {type_diff['deleted']}")
                if "added" in type_diff:
                    issue_details.append(f"Type added: {type_diff['added']}")

            # Check for nullable issues
            if "listOfTypes" in prop_diff:
                lot_diff = prop_diff["listOfTypes"]
                if "added" in lot_diff and "null" in lot_diff["added"]:
                    issue_details.append("Nullable added (OpenAI non-nullable)")
                if "deleted" in lot_diff and "null" in lot_diff["deleted"]:
                    issue_details.append("Nullable removed (OpenAI nullable)")

            # Check for union issues
            if "anyOf" in prop_diff or "oneOf" in prop_diff:
                union_diff = prop_diff.get("anyOf", prop_diff.get("oneOf", {}))
                if "added" in union_diff:
                    issue_details.append(f"Union variants added: {len(union_diff['added'])}")
                if "deleted" in union_diff:
                    issue_details.append(f"Union variants removed: {len(union_diff['deleted'])}")

            # Check for default value issues
            if "default" in prop_diff:
                default_diff = prop_diff["default"]
                if isinstance(default_diff, dict) and "from" in default_diff:
                    issue_details.append(f"Default changed: {default_diff['from']} -> {default_diff.get('to')}")

            if issue_details:
                result["issues"].append({"property": prop_path, "details": issue_details})

            # Recurse into nested properties
            nested = _extract_issues(prop_diff, prop_path)
            result["missing"].extend(nested["missing"])
            result["issues"].extend(nested["issues"])

    # Recurse into schema containers
    for key in ["schema", "items", "properties", "content", "responses", "parameters", "requestBody"]:
        if key in obj and isinstance(obj[key], dict):
            nested_path = f"{path}.{key}" if path and key not in ("schema", "items") else path
            nested = _extract_issues(obj[key], nested_path)
            result["missing"].extend(nested["missing"])
            result["issues"].extend(nested["issues"])

    return result


def _merge_diffs(base_diff: dict[str, Any], overlay_diff: dict[str, Any], overlay_paths: set[str]) -> dict[str, Any]:
    """Merge two oasdiff outputs, replacing paths from overlay_paths with overlay_diff data."""
    result = json.loads(json.dumps(base_diff))  # Deep copy

    base_paths = result.get("paths", {})
    overlay_paths_data = overlay_diff.get("paths", {})

    # Remove overlay paths from base deleted list
    if "deleted" in base_paths:
        base_paths["deleted"] = [p for p in base_paths["deleted"] if p not in overlay_paths]

    # Add overlay deleted paths
    if "deleted" in overlay_paths_data:
        if "deleted" not in base_paths:
            base_paths["deleted"] = []
        base_paths["deleted"].extend(overlay_paths_data["deleted"])

    # Remove overlay paths from base modified
    if "modified" in base_paths:
        for path in list(base_paths["modified"].keys()):
            if path in overlay_paths:
                del base_paths["modified"][path]

    # Add overlay modified paths
    if "modified" in overlay_paths_data:
        if "modified" not in base_paths:
            base_paths["modified"] = {}
        base_paths["modified"].update(overlay_paths_data["modified"])

    result["paths"] = base_paths
    return result


def calculate_coverage(
    openai_spec: Path,
    llama_spec: Path,
    openresponses_spec: Path | None = None,
) -> dict[str, Any]:
    """Calculate coverage metrics."""
    # Load categories from OpenAI spec
    endpoint_categories = _load_endpoint_categories(openai_spec)

    # Run main diff
    diff = _run_oasdiff(openai_spec, llama_spec)

    # If openresponses spec is provided, use it for Responses category
    if openresponses_spec and openresponses_spec.exists():
        responses_diff = _run_oasdiff(openresponses_spec, llama_spec)
        # Get all paths that start with /responses
        responses_paths = set(endpoint_categories.get("Responses", []))
        # Also include any /responses paths from the openresponses spec
        responses_spec_data = json.loads(openresponses_spec.read_text())
        for path in responses_spec_data.get("paths", {}).keys():
            responses_paths.add(path)
        diff = _merge_diffs(diff, responses_diff, responses_paths)

    paths_diff = diff.get("paths", {})
    deleted_paths = paths_diff.get("deleted", [])
    modified_paths = paths_diff.get("modified", {})

    # Filter to relevant categories (endpoints that exist in OpenAI spec)
    missing_endpoints = sorted([p for p in deleted_paths if _categorize_path(p, endpoint_categories) != "Other"])
    implemented_endpoints = sorted(
        [p for p in modified_paths.keys() if _categorize_path(p, endpoint_categories) != "Other"]
    )

    # Analyze each endpoint
    categories: dict[str, dict[str, Any]] = {}

    for path in implemented_endpoints:
        category = _categorize_path(path, endpoint_categories)
        if category not in categories:
            categories[category] = {
                "endpoints": [],
                "total_issues": 0,
                "total_missing": 0,
            }

        endpoint_diff = modified_paths[path]
        operations = endpoint_diff.get("operations", {}).get("modified", {})

        endpoint_data = {
            "path": path,
            "operations": [],
        }

        for method in sorted(operations.keys()):
            op_diff = operations[method]
            issues_data = _extract_issues(op_diff, method)

            sorted_missing = sorted(issues_data["missing"])
            sorted_issues = sorted(issues_data["issues"], key=lambda x: x["property"])

            endpoint_data["operations"].append(
                {
                    "method": method,
                    "missing_properties": sorted_missing,
                    "conformance_issues": sorted_issues,
                    "missing_count": len(sorted_missing),
                    "issues_count": len(sorted_issues),
                }
            )

            categories[category]["total_issues"] += len(sorted_issues)
            categories[category]["total_missing"] += len(sorted_missing)

        categories[category]["endpoints"].append(endpoint_data)

    # Calculate totals
    total_issues = sum(c["total_issues"] for c in categories.values())
    total_missing = sum(c["total_missing"] for c in categories.values())
    total_endpoints = len(implemented_endpoints) + len(missing_endpoints)

    # Calculate scores
    # Total "checked" = issues + missing (things oasdiff found problems with)
    # Conformant = things that match (not in diff)
    # We estimate conformant based on: for every issue, assume there are ~3 conformant properties
    # This gives a reasonable baseline that increases as issues decrease
    total_problems = total_issues + total_missing
    # Use a baseline of ~500 total properties (estimated from OpenAI spec)
    estimated_total = max(total_problems + 200, 500)
    overall_score = round((1 - total_problems / estimated_total) * 100, 1)

    # Build report
    report: dict[str, Any] = {
        "openai_spec": str(openai_spec),
        "openresponses_spec": str(openresponses_spec) if openresponses_spec else None,
        "llama_spec": str(llama_spec),
        "summary": {
            "endpoints": {
                "implemented": len(implemented_endpoints),
                "total": total_endpoints,
                "missing": missing_endpoints,
            },
            "conformance": {
                "score": overall_score,
                "issues": total_issues,
                "missing_properties": total_missing,
                "total_problems": total_problems,
            },
        },
        "categories": {},
    }

    # Build category details with per-category scores
    for cat_name in sorted(categories.keys()):
        cat_data = categories[cat_name]
        cat_problems = cat_data["total_issues"] + cat_data["total_missing"]
        # Estimate category total based on problems + baseline per category
        cat_estimated_total = max(cat_problems + 20, 50)
        cat_score = round((1 - cat_problems / cat_estimated_total) * 100, 1)

        report["categories"][cat_name] = {
            "score": cat_score,
            "issues": cat_data["total_issues"],
            "missing_properties": cat_data["total_missing"],
            "endpoints": sorted(cat_data["endpoints"], key=lambda x: x["path"]),
        }

    return report


def print_summary(report: dict[str, Any]) -> None:
    """Print a human-readable summary."""
    summary = report["summary"]

    print("\n" + "=" * 60)
    print("OpenAI API Conformance Report")
    print("=" * 60)
    print()

    # Overall score
    conf = summary["conformance"]
    print(f"🎯 Overall Conformance Score: {conf['score']}%")
    print()

    # Endpoints
    ep = summary["endpoints"]
    print(f"📡 Endpoints: {ep['implemented']}/{ep['total']} implemented")
    if ep["missing"]:
        print(f"   Missing: {', '.join(ep['missing'][:3])}")
        if len(ep["missing"]) > 3:
            print(f"            ... and {len(ep['missing']) - 3} more")
    print()

    # Conformance issues
    print("🔍 Remaining Issues")
    print("-" * 40)
    print(f"   Schema/Type issues:    {conf['issues']}")
    print(f"   Missing properties:    {conf['missing_properties']}")
    print(f"   Total to fix:          {conf['total_problems']}")
    print()

    # Category breakdown with scores
    print("📂 Score by Category")
    print("-" * 50)
    print(f"{'Category':<20} {'Score':<10} {'Issues':<10} {'Missing':<10}")
    print("-" * 50)

    for cat_name in sorted(
        report["categories"].keys(),
        key=lambda x: report["categories"][x]["score"],
    ):
        cat = report["categories"][cat_name]
        print(f"{cat_name:<20} {cat['score']:>5}%    {cat['issues']:<10} {cat['missing_properties']:<10}")

    print()


def main():
    parser = argparse.ArgumentParser(description="OpenAI API conformance analyzer")
    parser.add_argument(
        "--openai-spec",
        type=Path,
        default=Path("docs/static/openai-spec-2.3.0.yml"),
        help="Path to OpenAI spec",
    )
    parser.add_argument(
        "--openresponses-spec",
        type=Path,
        default=Path("docs/static/openresponses-spec.json"),
        help="Path to OpenResponses spec (used for Responses API)",
    )
    parser.add_argument(
        "--llama-spec",
        type=Path,
        default=Path("docs/static/llama-stack-spec.yaml"),
        help="Path to Llama Stack spec",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/static/openai-coverage.json"),
        help="Output path for coverage JSON",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update the coverage file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only output errors",
    )
    parser.add_argument(
        "--generate-docs",
        action="store_true",
        help="Also generate documentation from coverage report",
    )

    args = parser.parse_args()

    try:
        report = calculate_coverage(args.openai_spec, args.llama_spec, args.openresponses_spec)
    except FileNotFoundError:
        print("Error: oasdiff not found. Install with: go install github.com/tufin/oasdiff@latest")
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not args.quiet:
        print_summary(report)

    if args.update:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        if not args.quiet:
            print(f"✅ Report written to {args.output}")

    if args.generate_docs:
        script_dir = Path(__file__).parent
        result = subprocess.run(
            [sys.executable, script_dir / "generate_openai_coverage_docs.py"],
            cwd=script_dir.parent,
        )
        if result.returncode != 0:
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()

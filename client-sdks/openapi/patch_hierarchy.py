#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Patch generated API classes to add hierarchical properties.

This is a post-generation step that reads the api-hierarchy.yml produced by
process_openapi_hierarchy.py and patches the generated Python SDK to wire up
parent-child API relationships, enabling nested access like:

    client.chat.completions.create(...)

For a hierarchy like {chat: {completions: {}}}, this will:
1. Add an import of CompletionsApi in chat_api.py
2. Add a property `self.completions` in ChatApi
"""

import argparse
import re
import sys
from pathlib import Path

try:
    import ruamel.yaml as ryaml
except ImportError:
    print("Error: ruamel.yaml is required. Install with: uv pip install ruamel.yaml", file=sys.stderr)
    sys.exit(1)


def to_snake_case(name: str) -> str:
    """Convert a tag name to snake_case (e.g., 'DatasetIO' -> 'dataset_io')."""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower().replace(" ", "_").replace("-", "_")


def to_pascal_case(name: str) -> str:
    """Convert a tag name to PascalCase (e.g., 'chat_completions' -> 'ChatCompletions')."""
    words = re.split(r"[_\-\s]+", name)
    return "".join(word.capitalize() for word in words)


def extract_parent_child_pairs(hierarchy: dict, parent: str | None = None) -> list[tuple[str, str]]:
    """Extract all (parent, child) pairs from a nested hierarchy dict."""
    pairs = []
    for key, value in hierarchy.items():
        if parent:
            pairs.append((parent, key))
        if value:
            pairs.extend(extract_parent_child_pairs(value, key))
    return pairs


def patch_api_file(api_file: Path, child_tag: str, package_name: str) -> bool:
    """Patch an API file to add a child API property.

    Adds an import and an `Optional[ChildApi]` attribute to the parent API class.

    Returns True if the file was modified.
    """
    if not api_file.exists():
        print(f"  Warning: {api_file} does not exist, skipping")
        return False

    with open(api_file) as f:
        lines = f.readlines()

    child_snake = to_snake_case(child_tag)
    child_pascal = to_pascal_case(child_tag)
    child_class = f"{child_pascal}Api"
    child_module = f"{child_snake}_api"

    import_line = f"from {package_name}.api.{child_module} import {child_class}\n"

    if any(import_line.strip() in line for line in lines):
        return False

    # Find class definition
    class_line_idx = None
    for i, line in enumerate(lines):
        if re.match(r"^class \w+Api:", line):
            class_line_idx = i
            break

    if class_line_idx is None:
        print(f"  Warning: No class definition found in {api_file}")
        return False

    # Insert import before class definition
    import_idx = max(0, class_line_idx - 2)
    lines.insert(import_idx, import_line)

    # Find "self.api_client = api_client" after class definition
    api_client_line_idx = None
    for i in range(class_line_idx + 1, len(lines)):
        if "self.api_client = api_client" in lines[i]:
            api_client_line_idx = i
            break

    if api_client_line_idx is None:
        print(f"  Warning: 'self.api_client = api_client' not found in {api_file}")
        return False

    indent = len(lines[api_client_line_idx]) - len(lines[api_client_line_idx].lstrip())
    property_line = f"{' ' * indent}self.{child_snake}: Optional[{child_class}] = None\n"
    lines.insert(api_client_line_idx + 1, property_line)

    with open(api_file, "w") as f:
        f.writelines(lines)

    print(f"  Patched: {api_file.name} (added {child_snake})")
    return True


def patch_optional_import(api_file: Path) -> bool:
    """Ensure `Optional` is imported from typing in the given file.

    Returns True if the import was added.
    """
    with open(api_file) as f:
        content = f.read()

    if re.search(r"from typing import.*Optional", content):
        return False

    typing_import_match = re.search(r"from typing import ([^\n]+)", content)
    if typing_import_match:
        current_imports = typing_import_match.group(1)
        if "Optional" not in current_imports:
            new_imports = current_imports.rstrip() + ", Optional"
            content = content.replace(f"from typing import {current_imports}", f"from typing import {new_imports}")
            with open(api_file, "w") as f:
                f.write(content)
            return True
    else:
        lines = content.split("\n")
        import_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                import_idx = i + 1
        lines.insert(import_idx, "from typing import Optional")
        with open(api_file, "w") as f:
            f.write("\n".join(lines))
        return True

    return False


def patch_apis(hierarchy_file: str, sdk_dir: str, package_name: str = "llama_stack_client") -> None:
    """Patch all generated API files based on the hierarchy."""
    yaml_handler = ryaml.YAML()
    yaml_handler.preserve_quotes = True

    print(f"Loading hierarchy from: {hierarchy_file}")
    with open(hierarchy_file) as f:
        data = yaml_handler.load(f)

    hierarchy = data.get("api_hierarchy", {})
    if not hierarchy:
        print("No hierarchy found, nothing to patch")
        return

    pairs = extract_parent_child_pairs(hierarchy)
    print(f"Found {len(pairs)} parent-child relationships")

    api_dir = Path(sdk_dir) / package_name / "api"
    if not api_dir.exists():
        print(f"Error: API directory not found: {api_dir}", file=sys.stderr)
        sys.exit(1)

    # Patch individual API files
    patched_count = 0
    for parent_tag, child_tag in pairs:
        parent_snake = to_snake_case(parent_tag)
        parent_file = api_dir / f"{parent_snake}_api.py"

        if parent_file.exists():
            patch_optional_import(parent_file)

        if patch_api_file(parent_file, child_tag, package_name):
            patched_count += 1

    print(f"Patched {patched_count} API files")


def main():
    parser = argparse.ArgumentParser(description="Patch generated API classes with hierarchical properties")
    parser.add_argument(
        "--hierarchy",
        "-H",
        default="api-hierarchy.yml",
        help="API hierarchy file (default: api-hierarchy.yml)",
    )
    parser.add_argument(
        "--sdk-dir",
        "-s",
        default="sdks/python",
        help="SDK directory (default: sdks/python)",
    )
    parser.add_argument(
        "--package",
        "-p",
        default="llama_stack_client",
        help="Package name (default: llama_stack_client)",
    )

    args = parser.parse_args()

    if not Path(args.hierarchy).exists():
        print(f"Error: Hierarchy file not found: {args.hierarchy}", file=sys.stderr)
        sys.exit(1)

    if not Path(args.sdk_dir).exists():
        print(f"Error: SDK directory not found: {args.sdk_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        patch_apis(args.hierarchy, args.sdk_dir, args.package)
    except Exception as e:
        print(f"Error patching API files: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""
FIPS Compliance Checker

This script scans all Python code within the llama_stack directory to check
for usage of the following FIPS-non-compliant cryptographic functions:
- hashlib.md5
- hashlib.sha1
- uuid.uuid3
- uuid.uuid5

Exit codes:
- 0: No prohibited functions found
- 1: One or more prohibited functions found
"""

import ast
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


class FIPSViolationFinder(ast.NodeVisitor):
    """AST visitor to find FIPS-non-compliant function calls."""

    def __init__(self):
        self.violations: list[tuple[str, int, str]] = []
        self.current_file = ""

        # Prohibited functions we're looking for
        self.prohibited_functions = {"hashlib.md5", "hashlib.sha1", "uuid.uuid3", "uuid.uuid5"}

        # Track imports to handle different import styles
        self.hashlib_imported = False
        self.uuid_imported = False
        self.hashlib_alias = None
        self.uuid_alias = None

    def visit_import(self, node):
        """Handle access case
        'import <module>'
        """
        for alias in node.names:
            if alias.name == "hashlib":
                self.hashlib_imported = True
                self.hashlib_alias = alias.asname or "hashlib"
            elif alias.name == "uuid":
                self.uuid_imported = True
                self.uuid_alias = alias.asname or "uuid"
        self.generic_visit(node)

    def visit_import_from(self, node):
        """Handle access case
        'from <module> import <function>'
        """
        if node.module == "hashlib":
            for alias in node.names:
                if alias.name in ["md5", "sha1"]:
                    violation = f"from hashlib import {alias.name}"
                    if alias.asname:
                        violation += f" as {alias.asname}"
                    self.violations.append((self.current_file, node.lineno, violation))
        elif node.module == "uuid":
            for alias in node.names:
                if alias.name in ["uuid3", "uuid5"]:
                    violation = f"from uuid import {alias.name}"
                    if alias.asname:
                        violation += f" as {alias.asname}"
                    self.violations.append((self.current_file, node.lineno, violation))
        self.generic_visit(node)

    def visit_attribute(self, node):
        """Handle access case
        '<module>.<function>'
        """
        if isinstance(node.value, ast.Name):
            # Direct module.function access
            module_name = node.value.id
            attr_name = node.attr
            # Check for hashlib violations
            if (module_name == "hashlib" or module_name == self.hashlib_alias) and attr_name in ["md5", "sha1"]:
                violation = f"{module_name}.{attr_name}"
                self.violations.append((self.current_file, node.lineno, violation))
            # Check for uuid violations
            elif (module_name == "uuid" or module_name == self.uuid_alias) and attr_name in ["uuid3", "uuid5"]:
                violation = f"{module_name}.{attr_name}"
                self.violations.append((self.current_file, node.lineno, violation))
        self.generic_visit(node)

    def visit_call(self, node):
        """Handle function calls to catch direct calls to prohibited functions."""
        if isinstance(node.func, ast.Attribute):
            self.visit_attribute(node.func)
        elif isinstance(node.func, ast.Name):
            # Handle cases where functions were imported directly
            func_name = node.func.id
            if func_name in ["md5", "sha1", "uuid3", "uuid5"]:
                # This could be a prohibited function if imported directly
                violation = f"{func_name}()"
                self.violations.append((self.current_file, node.lineno, violation))

        self.generic_visit(node)


def scan_python_file(file_path: Path) -> list[tuple[str, int, str]]:
    """Scan a single Python file for FIPS violations."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # Parse the AST
        tree = ast.parse(content, filename=str(file_path))

        # Find violations
        finder = FIPSViolationFinder()
        finder.current_file = str(file_path)
        finder.visit(tree)

        return finder.violations

    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"Error: Could not parse {file_path}: {e}", file=sys.stderr)
        sys.exit(1)


def find_python_files(directory: Path) -> list[Path]:
    """Find all Python files in the given directory."""
    python_files = []

    for root, dirs, files in os.walk(directory):
        # Skip common non-source directories
        dirs[:] = [
            d
            for d in dirs
            if d not in {".git", "__pycache__", ".pytest_cache", "node_modules", ".venv", "venv", ".tox"}
        ]

        for file in files:
            if file.endswith(".py"):
                python_files.append(Path(root) / file)

    return python_files


def main():
    """Main function to scan for FIPS violations."""

    llama_stack_dir = REPO_ROOT / "llama_stack"
    if not llama_stack_dir.exists():
        print(f"Error: llama_stack directory not found at {llama_stack_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning Python files in {llama_stack_dir} for FIPS violations...")

    # Find all Python files
    python_files = find_python_files(llama_stack_dir)
    print(f"Found {len(python_files)} Python files to scan")

    # Scan each file
    all_violations = []
    for file_path in python_files:
        violations = scan_python_file(file_path)
        all_violations.extend(violations)

    # Report results
    if all_violations:
        print("\n❌ FIPS COMPLIANCE CHECK FAILED")
        print(f"Found {len(all_violations)} violation(s):\n")

        for file_path, line_num, violation in all_violations:
            # Make path relative to project root for cleaner output
            rel_path = Path(file_path).relative_to(REPO_ROOT)
            print(f"  {rel_path}:{line_num} - {violation}")

        print("\nProhibited functions found:")
        print("  - hashlib.md5 and hashlib.sha1 are not FIPS-compliant")
        print("  - uuid.uuid3 and uuid.uuid5 use MD5 and SHA-1 respectively")
        print("  - Consider using hashlib.sha256, hashlib.sha384, or hashlib.sha512")
        print("  - Consider using uuid.uuid4 (random) or uuid.uuid1 (MAC-based)")
        sys.exit(1)
    else:
        print("\n✅ FIPS COMPLIANCE CHECK PASSED")
        print(f"No prohibited cryptographic functions found in {len(python_files)} files")
        sys.exit(0)


if __name__ == "__main__":
    main()

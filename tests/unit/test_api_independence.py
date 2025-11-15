# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Test that llama_stack_api does not depend on llama_stack.

Ensures the API package remains independent and can be published separately.
"""

import ast
import pathlib
import tomllib


def get_imports(file_path: pathlib.Path) -> set[str]:
    """Extract top-level module names imported in a Python file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(file_path))

        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])

        return imports
    except (SyntaxError, UnicodeDecodeError):
        return set()


def test_api_package_independence():
    """Verify llama_stack_api does not depend on llama_stack."""
    project_root = pathlib.Path(__file__).resolve().parent.parent.parent
    api_path = project_root / "src" / "llama_stack_api"

    if not api_path.exists():
        return

    errors = []

    # Check 1: No imports of llama_stack in code
    import_violations = []
    for py_file in api_path.rglob("*.py"):
        if "llama_stack" in get_imports(py_file):
            with open(py_file, encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    stripped = line.strip()
                    if (
                        stripped.startswith(("import llama_stack", "from llama_stack"))
                        and "llama_stack_api" not in stripped
                    ):
                        rel_path = py_file.relative_to(project_root)
                        import_violations.append(f"{rel_path}:{i}: {line.strip()}")

    if import_violations:
        errors.append("Import violations found in code:")
        errors.extend(f"  {v}" for v in import_violations)

    # Check 2: No llama_stack in pyproject.toml dependencies
    pyproject_path = api_path / "pyproject.toml"
    if pyproject_path.exists():
        with open(pyproject_path, "rb") as f:
            pyproject = tomllib.load(f)

        dependencies = pyproject.get("project", {}).get("dependencies", [])
        dep_violations = [dep for dep in dependencies if "llama_stack" in dep and "llama_stack_api" not in dep]

        if dep_violations:
            errors.append("\nDependency violations in pyproject.toml:")
            errors.extend(f"  {dep}" for dep in dep_violations)

    if errors:
        error_msg = [
            "\nllama_stack_api must not depend on llama_stack!",
            "\nViolations:",
        ]
        error_msg.extend(errors)
        error_msg.append("\nThe API package must remain independent for separate publishing.")
        raise AssertionError("\n".join(error_msg))

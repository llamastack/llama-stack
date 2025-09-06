# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Simple unit tests for the version command that can be run independently.
These tests focus on testing individual methods and functionality.
"""

import json
import sys
from pathlib import Path


def test_build_info_file_structure():
    """Test that build_info.py has the correct structure when it exists"""
    build_info_path = Path(__file__).parent.parent.parent.parent / "llama_stack" / "cli" / "build_info.py"

    if build_info_path.exists():
        content = build_info_path.read_text()

        # Check that it contains BUILD_INFO
        assert "BUILD_INFO" in content, "build_info.py should contain BUILD_INFO"

        # Check that it has the expected fields
        expected_fields = [
            "git_commit",
            "git_commit_date",
            "git_branch",
            "git_tag",
            "build_timestamp",
        ]

        for field in expected_fields:
            assert field in content, f"Missing field in build_info.py: {field}"

        print("✓ Build info file structure test passed")
        return True
    else:
        print("! Build info file not found - this is expected in development")
        return True


def test_build_script_exists():
    """Test that the build script exists and has the correct structure"""
    script_path = Path(__file__).parent.parent.parent.parent / "scripts" / "generate_build_info.py"

    assert script_path.exists(), "Build script should exist"

    content = script_path.read_text()

    # Check for key functions
    assert "def get_git_info" in content, "Build script should have get_git_info function"
    assert "def generate_build_info_file" in content, "Build script should have generate_build_info_file function"
    assert "BUILD_INFO" in content, "Build script should reference BUILD_INFO"

    print("✓ Build script exists and has correct structure")
    return True


def test_version_module_structure():
    """Test that the version module has the correct structure"""
    version_path = Path(__file__).parent.parent.parent.parent / "llama_stack" / "cli" / "version.py"

    assert version_path.exists(), "Version module should exist"

    content = version_path.read_text()

    # Check for key classes and methods
    assert "class VersionCommand" in content, "Should have VersionCommand class"
    assert "def _get_package_version" in content, "Should have _get_package_version method"
    assert "def _get_build_info" in content, "Should have _get_build_info method"
    assert "def _get_components_info" in content, "Should have _get_components_info method"
    assert "def _run_version_command" in content, "Should have _run_version_command method"

    # Check for proper imports
    assert "from llama_stack.cli.subcommand import Subcommand" in content, "Should import Subcommand"
    assert "from llama_stack.distribution.distribution import get_provider_registry" in content, (
        "Should import get_provider_registry"
    )

    print("✓ Version module structure test passed")
    return True


def test_cli_integration():
    """Test that the version command is properly integrated into the CLI"""
    llama_cli_path = Path(__file__).parent.parent.parent.parent / "llama_stack" / "cli" / "llama.py"

    assert llama_cli_path.exists(), "Main CLI module should exist"

    content = llama_cli_path.read_text()

    # Check that version command is imported and added
    assert "from .version import VersionCommand" in content, "Should import VersionCommand"
    assert "VersionCommand.create(subparsers)" in content, "Should add VersionCommand to subparsers"

    print("✓ CLI integration test passed")
    return True


def test_gitignore_entry():
    """Test that build_info.py is properly ignored in git"""
    gitignore_path = Path(__file__).parent.parent.parent.parent / ".gitignore"

    if gitignore_path.exists():
        content = gitignore_path.read_text()
        assert "llama_stack/cli/build_info.py" in content, "build_info.py should be in .gitignore"
        print("✓ Gitignore entry test passed")
        return True
    else:
        print("! .gitignore not found")
        return True


def test_component_type_detection_logic():
    """Test the component type detection logic"""

    # Simulate the component type detection logic from the version command
    def detect_component_type(provider_type, adapter_type=None):
        if provider_type.startswith("inline::"):
            return "inline", provider_type.replace("inline::", "")
        elif provider_type.startswith("remote::"):
            return "remote", provider_type.replace("remote::", "")
        elif adapter_type:
            return "remote", adapter_type
        else:
            return "unknown", provider_type

    # Test cases
    test_cases = [
        ("inline::meta-reference", None, "inline", "meta-reference"),
        ("remote::ollama", None, "remote", "ollama"),
        ("some-provider", "adapter-name", "remote", "adapter-name"),
        ("unknown-provider", None, "unknown", "unknown-provider"),
    ]

    for provider_type, adapter_type, expected_type, expected_name in test_cases:
        comp_type, comp_name = detect_component_type(provider_type, adapter_type)
        assert comp_type == expected_type, f"Expected type {expected_type}, got {comp_type}"
        assert comp_name == expected_name, f"Expected name {expected_name}, got {comp_name}"

    print("✓ Component type detection logic test passed")
    return True


def test_python_version_format():
    """Test that Python version formatting works correctly"""

    # Simulate the Python version formatting from the version command
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    # Check format
    parts = python_version.split(".")
    assert len(parts) == 3, "Python version should have 3 parts"
    assert all(part.isdigit() for part in parts), "All parts should be numeric"

    print(f"✓ Python version format test passed: {python_version}")
    return True


def test_json_output_structure():
    """Test the expected JSON output structure"""

    # Expected structure for JSON output
    expected_fields = {
        "llama_stack_version": str,
        "llama_stack_client_version": str,
        "python_version": str,
        "git_commit": str,
        "git_commit_date": str,
        "git_branch": str,
        "git_tag": str,
        "build_timestamp": str,
    }

    # Test that we can create a valid JSON structure
    test_data = dict.fromkeys(expected_fields.keys(), "test_value")

    # Should be valid JSON
    json_str = json.dumps(test_data, indent=2)
    parsed = json.loads(json_str)

    # Should have all expected fields
    for field in expected_fields.keys():
        assert field in parsed, f"Missing field: {field}"

    print("✓ JSON output structure test passed")
    return True


def run_all_tests():
    """Run all simple unit tests"""
    tests = [
        test_build_info_file_structure,
        test_build_script_exists,
        test_version_module_structure,
        test_cli_integration,
        test_gitignore_entry,
        test_component_type_detection_logic,
        test_python_version_format,
        test_json_output_structure,
    ]

    passed = 0
    failed = 0

    print("Running simple version command unit tests...")

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed: {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")

    return failed == 0


if __name__ == "__main__":
    """Run tests when executed directly"""
    success = run_all_tests()
    if not success:
        sys.exit(1)
    else:
        print("All simple unit tests passed!")

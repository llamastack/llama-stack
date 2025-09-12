# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Integration tests for the version command that can be run independently.
These tests verify the basic functionality without requiring complex mocking.
"""

import json
import subprocess
import sys
from pathlib import Path


def test_version_command_help():
    """Test that the version command help works"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "llama_stack.cli.llama", "version", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=Path(__file__).parent.parent.parent.parent,
        )

        assert result.returncode == 0
        assert "Display version information" in result.stdout
        assert "--format" in result.stdout
        assert "--components" in result.stdout
        print("✓ Version command help test passed")

    except subprocess.TimeoutExpired:
        print("✗ Version command help test timed out")
        raise
    except Exception as e:
        print(f"✗ Version command help test failed: {e}")
        raise


def test_version_command_basic():
    """Test basic version command execution"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "llama_stack.cli.llama", "version", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=Path(__file__).parent.parent.parent.parent,
        )

        assert result.returncode == 0

        # Parse JSON output
        json_output = json.loads(result.stdout.split("\n")[-2])  # Get last non-empty line

        # Check required fields
        required_fields = [
            "llama_stack_version",
            "llama_stack_client_version",
            "python_version",
            "git_commit",
            "git_commit_date",
            "git_branch",
            "git_tag",
            "build_timestamp",
        ]

        for field in required_fields:
            assert field in json_output, f"Missing field: {field}"

        # Check that values are not empty (except for potentially unknown values)
        assert json_output["python_version"] != ""
        assert "." in json_output["python_version"]  # Should be in format x.y.z

        print("✓ Version command basic test passed")
        print(f"  Llama Stack version: {json_output['llama_stack_version']}")
        print(f"  Python version: {json_output['python_version']}")
        print(f"  Git commit: {json_output['git_commit']}")

    except subprocess.TimeoutExpired:
        print("✗ Version command basic test timed out")
        raise
    except Exception as e:
        print(f"✗ Version command basic test failed: {e}")
        raise


def test_version_command_with_components():
    """Test version command with components flag"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "llama_stack.cli.llama", "version", "--format", "json", "--components"],
            capture_output=True,
            text=True,
            timeout=60,  # Longer timeout for components
            cwd=Path(__file__).parent.parent.parent.parent,
        )

        assert result.returncode == 0

        # Parse JSON output
        json_output = json.loads(result.stdout.split("\n")[-2])  # Get last non-empty line

        # Check that components field exists
        assert "components" in json_output
        assert isinstance(json_output["components"], list)

        # If components exist, check their structure
        if json_output["components"]:
            component = json_output["components"][0]
            required_component_fields = ["api", "component", "type", "provider_type"]
            for field in required_component_fields:
                assert field in component, f"Missing component field: {field}"

        print("✓ Version command with components test passed")
        print(f"  Found {len(json_output['components'])} components")

    except subprocess.TimeoutExpired:
        print("✗ Version command with components test timed out")
        raise
    except Exception as e:
        print(f"✗ Version command with components test failed: {e}")
        raise


def test_build_info_structure():
    """Test that build_info.py has the correct structure"""
    try:
        # Import build info directly
        build_info_path = Path(__file__).parent.parent.parent.parent / "llama_stack" / "cli" / "build_info.py"

        if build_info_path.exists():
            # Read the file content
            content = build_info_path.read_text()

            # Check that it contains BUILD_INFO
            assert "BUILD_INFO" in content

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

            print("✓ Build info structure test passed")
        else:
            print("! Build info file not found - this is expected in development")

    except Exception as e:
        print(f"✗ Build info structure test failed: {e}")
        raise


def test_build_script_execution():
    """Test that the build script can be executed"""
    try:
        script_path = Path(__file__).parent.parent.parent.parent / "scripts" / "generate_build_info.py"

        if script_path.exists():
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=script_path.parent.parent,
            )

            assert result.returncode == 0
            assert "Generated build info file" in result.stdout

            print("✓ Build script execution test passed")
        else:
            print("! Build script not found")

    except subprocess.TimeoutExpired:
        print("✗ Build script execution test timed out")
        raise
    except Exception as e:
        print(f"✗ Build script execution test failed: {e}")
        raise


if __name__ == "__main__":
    """Run integration tests when executed directly"""
    print("Running version command integration tests...")

    tests = [
        test_version_command_help,
        test_version_command_basic,
        test_version_command_with_components,
        test_build_info_structure,
        test_build_script_execution,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"Test {test.__name__} failed: {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")

    if failed > 0:
        sys.exit(1)
    else:
        print("All integration tests passed!")

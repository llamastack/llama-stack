# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import json
from unittest.mock import patch

import pytest

from llama_stack.cli.version import VersionCommand


class TestVersionCommand:
    """Test suite for the VersionCommand class"""

    @pytest.fixture
    def version_command(self):
        """Create a VersionCommand instance for testing"""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        return VersionCommand(subparsers)

    def test_version_command_basic_functionality(self, version_command):
        """Test basic version command functionality"""
        # Test package version retrieval
        with patch("llama_stack.cli.version.version") as mock_version:
            mock_version.return_value = "0.2.12"
            assert version_command._get_package_version("llama-stack") == "0.2.12"

            # Test missing package
            from importlib.metadata import PackageNotFoundError

            mock_version.side_effect = PackageNotFoundError()
            assert version_command._get_package_version("missing-package") == "unknown"

        # Test build info with mocked BUILD_INFO
        mock_build_info = {
            "git_commit": "abc123",
            "git_commit_date": "2025-01-15",
            "git_branch": "main",
            "git_tag": "v0.2.12",
            "build_timestamp": "2025-01-15T18:30:00+00:00",
        }
        with patch("llama_stack.cli.version.BUILD_INFO", mock_build_info):
            result = version_command._get_build_info()
            assert result["commit_hash"] == "abc123"
            assert result["branch"] == "main"

        # Test default JSON output (should only show version)
        args_default = argparse.Namespace(output="json", build_info=False, dependencies=False, all=False)
        with (
            patch.object(version_command, "_get_package_version") as mock_get_version,
            patch("builtins.print") as mock_print,
        ):
            mock_get_version.return_value = "0.2.12"

            version_command._run_version_command(args_default)

            printed_output = mock_print.call_args[0][0]
            json_output = json.loads(printed_output)
            assert json_output["llama_stack_version"] == "0.2.12"
            # Should not include build info in default output
            assert "git_commit" not in json_output

        # Test JSON output with build info
        args_with_build_info = argparse.Namespace(output="json", build_info=True, dependencies=False, all=False)
        with (
            patch.object(version_command, "_get_package_version") as mock_get_version,
            patch.object(version_command, "_get_build_info") as mock_get_build_info,
            patch("builtins.print") as mock_print,
        ):
            mock_get_version.return_value = "0.2.12"
            mock_get_build_info.return_value = {
                "commit_hash": "abc123",
                "commit_date": "2025-01-15",
                "branch": "main",
                "tag": "v0.2.12",
                "build_timestamp": "2025-01-15T18:30:00+00:00",
            }

            version_command._run_version_command(args_with_build_info)

            printed_output = mock_print.call_args[0][0]
            json_output = json.loads(printed_output)
            assert json_output["llama_stack_version"] == "0.2.12"
            assert json_output["git_commit"] == "abc123"

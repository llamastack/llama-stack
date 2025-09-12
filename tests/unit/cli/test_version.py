# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import json
import sys
from io import StringIO
from unittest.mock import Mock, patch

import pytest

from llama_stack.cli.version import VersionCommand


class TestVersionCommand:
    """Test suite for the VersionCommand class"""

    @pytest.fixture
    def version_command(self):
        """Create a VersionCommand instance for testing"""
        # Create a mock subparsers object
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        return VersionCommand(subparsers)

    @pytest.fixture
    def mock_build_info(self):
        """Mock build info data"""
        return {
            "git_commit": "abc123def456",
            "git_commit_date": "2025-01-15 10:30:00 -0800",
            "git_branch": "main",
            "git_tag": "v0.2.12",
            "build_timestamp": "2025-01-15T18:30:00.123456+00:00",
        }

    @pytest.fixture
    def mock_components_info(self):
        """Mock components info data"""
        return [
            {
                "api": "inference",
                "component": "meta-reference",
                "type": "inline",
                "provider_type": "inline::meta-reference",
            },
            {
                "api": "inference",
                "component": "vllm",
                "type": "inline",
                "provider_type": "inline::vllm",
            },
            {
                "api": "inference",
                "component": "ollama",
                "type": "remote",
                "provider_type": "remote::ollama",
            },
            {
                "api": "safety",
                "component": "llama-guard",
                "type": "inline",
                "provider_type": "inline::llama-guard",
            },
        ]

    def test_get_package_version_existing_package(self, version_command):
        """Test getting version of an existing package"""
        with patch("llama_stack.cli.version.version") as mock_version:
            mock_version.return_value = "1.2.3"
            result = version_command._get_package_version("test-package")
            assert result == "1.2.3"
            mock_version.assert_called_once_with("test-package")

    def test_get_package_version_missing_package(self, version_command):
        """Test getting version of a non-existent package"""
        with patch("llama_stack.cli.version.version") as mock_version:
            from importlib.metadata import PackageNotFoundError

            mock_version.side_effect = PackageNotFoundError()
            result = version_command._get_package_version("non-existent-package")
            assert result == "unknown"

    def test_get_build_info_with_build_info_module(self, version_command, mock_build_info):
        """Test getting build info when build_info module is available"""
        mock_build_info_dict = {
            "git_commit": mock_build_info["git_commit"],
            "git_commit_date": mock_build_info["git_commit_date"],
            "git_branch": mock_build_info["git_branch"],
            "git_tag": mock_build_info["git_tag"],
            "build_timestamp": mock_build_info["build_timestamp"],
        }

        with patch("llama_stack.cli.version.BUILD_INFO", mock_build_info_dict):
            result = version_command._get_build_info()
            assert result["commit_hash"] == mock_build_info["git_commit"]
            assert result["commit_date"] == mock_build_info["git_commit_date"]
            assert result["branch"] == mock_build_info["git_branch"]
            assert result["tag"] == mock_build_info["git_tag"]
            assert result["build_timestamp"] == mock_build_info["build_timestamp"]

    def test_get_build_info_without_build_info_module(self, version_command):
        """Test getting build info when build_info module is not available"""
        with patch("llama_stack.cli.version.BUILD_INFO", side_effect=ImportError()):
            result = version_command._get_build_info()
            assert result["commit_hash"] == "unknown"
            assert result["commit_date"] == "unknown"
            assert result["branch"] == "unknown"
            assert result["tag"] == "unknown"
            assert result["build_timestamp"] == "unknown"

    def test_get_components_info_success(self, version_command, mock_components_info):
        """Test getting components info successfully"""
        # Mock the provider registry
        mock_registry = {
            "inference": {
                "inline::meta-reference": Mock(
                    api=Mock(value="inference"),
                    provider_type="inline::meta-reference",
                    adapter_type=None,
                ),
                "inline::vllm": Mock(
                    api=Mock(value="inference"),
                    provider_type="inline::vllm",
                    adapter_type=None,
                ),
                "remote::ollama": Mock(
                    api=Mock(value="inference"),
                    provider_type="remote::ollama",
                    adapter_type="ollama",
                ),
            },
            "safety": {
                "inline::llama-guard": Mock(
                    api=Mock(value="safety"),
                    provider_type="inline::llama-guard",
                    adapter_type=None,
                ),
            },
        }

        with patch("llama_stack.cli.version.get_provider_registry") as mock_get_registry:
            mock_get_registry.return_value = mock_registry
            result = version_command._get_components_info()

            # Should have 4 components
            assert len(result) == 4

            # Check that all expected components are present
            component_names = [comp["component"] for comp in result]
            assert "meta-reference" in component_names
            assert "vllm" in component_names
            assert "ollama" in component_names
            assert "llama-guard" in component_names

    def test_get_components_info_exception(self, version_command):
        """Test getting components info when an exception occurs"""
        with patch("llama_stack.cli.version.get_provider_registry") as mock_get_registry:
            mock_get_registry.side_effect = Exception("Test error")

            # Capture stderr to check warning message
            with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
                result = version_command._get_components_info()
                assert result == []
                assert "Warning: Could not load components information" in mock_stderr.getvalue()

    def test_run_version_command_table_format(self, version_command, mock_build_info):
        """Test running version command with table format"""
        args = argparse.Namespace(format="table", components=False)

        with (
            patch.object(version_command, "_get_package_version") as mock_get_version,
            patch.object(version_command, "_get_build_info") as mock_get_build_info,
            patch("llama_stack.cli.version.print_table") as mock_print_table,
            patch("builtins.print") as mock_print,
        ):
            mock_get_version.side_effect = lambda pkg: {
                "llama-stack": "0.2.12",
                "llama-stack-client": "0.2.12",
            }.get(pkg, "unknown")

            mock_get_build_info.return_value = {
                "commit_hash": mock_build_info["git_commit"],
                "commit_date": mock_build_info["git_commit_date"],
                "branch": mock_build_info["git_branch"],
                "tag": mock_build_info["git_tag"],
                "build_timestamp": mock_build_info["build_timestamp"],
            }

            version_command._run_version_command(args)

            # Check that print was called with headers
            mock_print.assert_any_call("Llama Stack Version Information")
            mock_print.assert_any_call("=" * 50)
            mock_print.assert_any_call("\nBuild Information")
            mock_print.assert_any_call("-" * 30)

            # Check that print_table was called twice (version and build info)
            assert mock_print_table.call_count == 2

    def test_run_version_command_json_format(self, version_command, mock_build_info):
        """Test running version command with JSON format"""
        args = argparse.Namespace(format="json", components=False)

        with (
            patch.object(version_command, "_get_package_version") as mock_get_version,
            patch.object(version_command, "_get_build_info") as mock_get_build_info,
            patch("builtins.print") as mock_print,
        ):
            mock_get_version.side_effect = lambda pkg: {
                "llama-stack": "0.2.12",
                "llama-stack-client": "0.2.12",
            }.get(pkg, "unknown")

            mock_get_build_info.return_value = {
                "commit_hash": mock_build_info["git_commit"],
                "commit_date": mock_build_info["git_commit_date"],
                "branch": mock_build_info["git_branch"],
                "tag": mock_build_info["git_tag"],
                "build_timestamp": mock_build_info["build_timestamp"],
            }

            version_command._run_version_command(args)

            # Check that JSON was printed
            mock_print.assert_called_once()
            printed_output = mock_print.call_args[0][0]

            # Parse the JSON to verify it's valid and contains expected fields
            json_output = json.loads(printed_output)
            assert json_output["llama_stack_version"] == "0.2.12"
            assert json_output["llama_stack_client_version"] == "0.2.12"
            assert json_output["git_commit"] == mock_build_info["git_commit"]
            assert json_output["git_branch"] == mock_build_info["git_branch"]
            assert json_output["build_timestamp"] == mock_build_info["build_timestamp"]

    def test_run_version_command_with_components_table(self, version_command, mock_build_info, mock_components_info):
        """Test running version command with components in table format"""
        args = argparse.Namespace(format="table", components=True)

        with (
            patch.object(version_command, "_get_package_version") as mock_get_version,
            patch.object(version_command, "_get_build_info") as mock_get_build_info,
            patch.object(version_command, "_get_components_info") as mock_get_components_info,
            patch("builtins.print") as mock_print,
        ):
            mock_get_version.side_effect = lambda pkg: {
                "llama-stack": "0.2.12",
                "llama-stack-client": "0.2.12",
            }.get(pkg, "unknown")

            mock_get_build_info.return_value = {
                "commit_hash": mock_build_info["git_commit"],
                "commit_date": mock_build_info["git_commit_date"],
                "branch": mock_build_info["git_branch"],
                "tag": mock_build_info["git_tag"],
                "build_timestamp": mock_build_info["build_timestamp"],
            }

            mock_get_components_info.return_value = mock_components_info

            version_command._run_version_command(args)

            # Check that components section was printed
            mock_print.assert_any_call("\nAvailable Components")
            mock_print.assert_any_call("-" * 30)

            # Check that API sections were printed
            mock_print.assert_any_call("\nINFERENCE API:")
            mock_print.assert_any_call("\nSAFETY API:")

    def test_run_version_command_with_components_json(self, version_command, mock_build_info, mock_components_info):
        """Test running version command with components in JSON format"""
        args = argparse.Namespace(format="json", components=True)

        with (
            patch.object(version_command, "_get_package_version") as mock_get_version,
            patch.object(version_command, "_get_build_info") as mock_get_build_info,
            patch.object(version_command, "_get_components_info") as mock_get_components_info,
            patch("builtins.print") as mock_print,
        ):
            mock_get_version.side_effect = lambda pkg: {
                "llama-stack": "0.2.12",
                "llama-stack-client": "0.2.12",
            }.get(pkg, "unknown")

            mock_get_build_info.return_value = {
                "commit_hash": mock_build_info["git_commit"],
                "commit_date": mock_build_info["git_commit_date"],
                "branch": mock_build_info["git_branch"],
                "tag": mock_build_info["git_tag"],
                "build_timestamp": mock_build_info["build_timestamp"],
            }

            mock_get_components_info.return_value = mock_components_info

            version_command._run_version_command(args)

            # Check that JSON was printed
            mock_print.assert_called_once()
            printed_output = mock_print.call_args[0][0]

            # Parse the JSON to verify it contains components
            json_output = json.loads(printed_output)
            assert "components" in json_output
            assert len(json_output["components"]) == 4

            # Check that components have expected structure
            component = json_output["components"][0]
            assert "api" in component
            assert "component" in component
            assert "type" in component
            assert "provider_type" in component

    def test_run_version_command_no_components_available(self, version_command, mock_build_info):
        """Test running version command when no components are available"""
        args = argparse.Namespace(format="table", components=True)

        with (
            patch.object(version_command, "_get_package_version") as mock_get_version,
            patch.object(version_command, "_get_build_info") as mock_get_build_info,
            patch.object(version_command, "_get_components_info") as mock_get_components_info,
            patch("builtins.print") as mock_print,
        ):
            mock_get_version.side_effect = lambda pkg: {
                "llama-stack": "0.2.12",
                "llama-stack-client": "0.2.12",
            }.get(pkg, "unknown")

            mock_get_build_info.return_value = {
                "commit_hash": mock_build_info["git_commit"],
                "commit_date": mock_build_info["git_commit_date"],
                "branch": mock_build_info["git_branch"],
                "tag": mock_build_info["git_tag"],
                "build_timestamp": mock_build_info["build_timestamp"],
            }

            mock_get_components_info.return_value = []

            version_command._run_version_command(args)

            # Check that "no components" message was printed
            mock_print.assert_any_call("No components information available")

    def test_python_version_format(self, version_command):
        """Test that Python version is formatted correctly"""
        args = argparse.Namespace(format="json", components=False)

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

            version_command._run_version_command(args)

            printed_output = mock_print.call_args[0][0]
            json_output = json.loads(printed_output)

            # Check that Python version matches current Python version
            expected_python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            assert json_output["python_version"] == expected_python_version

    def test_component_type_detection(self, version_command):
        """Test that component types are detected correctly"""
        # Test inline provider
        mock_registry = {
            "inference": {
                "inline::test": Mock(
                    api=Mock(value="inference"),
                    provider_type="inline::test",
                    adapter_type=None,
                ),
                "remote::test": Mock(
                    api=Mock(value="inference"),
                    provider_type="remote::test",
                    adapter_type=None,
                ),
                "adapter-test": Mock(
                    api=Mock(value="inference"),
                    provider_type="adapter-test",
                    adapter_type="test-adapter",
                ),
            },
        }

        with patch("llama_stack.cli.version.get_provider_registry") as mock_get_registry:
            mock_get_registry.return_value = mock_registry
            result = version_command._get_components_info()

            # Find components by provider type to avoid conflicts
            inline_components = [comp for comp in result if comp["provider_type"] == "inline::test"]
            remote_components = [comp for comp in result if comp["provider_type"] == "remote::test"]
            adapter_components = [comp for comp in result if comp["provider_type"] == "adapter-test"]

            assert len(inline_components) == 1
            assert inline_components[0]["type"] == "inline"

            assert len(remote_components) == 1
            assert remote_components[0]["type"] == "remote"

            assert len(adapter_components) == 1
            assert adapter_components[0]["type"] == "remote"

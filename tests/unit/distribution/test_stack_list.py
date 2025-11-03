# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Tests for the llama stack list command.

These tests verify the fix for issue #3922 where `llama stack list` only showed
distributions after they were run.
"""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from llama_stack.cli.stack.list_stacks import StackListBuilds


@pytest.fixture
def list_stacks_command():
    """Create a StackListBuilds instance for testing."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    return StackListBuilds(subparsers)


@pytest.fixture
def mock_distribs_base_dir(tmp_path):
    """Create a mock DISTRIBS_BASE_DIR with some built distributions."""
    built_dir = tmp_path / "distributions"
    built_dir.mkdir(parents=True, exist_ok=True)

    # Create a built distribution
    starter_built = built_dir / "starter"
    starter_built.mkdir()
    (starter_built / "starter-build.yaml").write_text("# build config")
    (starter_built / "starter-run.yaml").write_text("# run config")

    return built_dir


@pytest.fixture
def mock_distro_dir(tmp_path):
    """Create a mock distributions directory with built-in distributions."""
    distro_dir = tmp_path / "src" / "llama_stack" / "distributions"
    distro_dir.mkdir(parents=True, exist_ok=True)

    # Create some built-in distributions
    for distro_name in ["starter", "nvidia", "dell"]:
        distro_path = distro_dir / distro_name
        distro_path.mkdir()
        (distro_path / "build.yaml").write_text("# build config")
        (distro_path / "run.yaml").write_text("# run config")

    return distro_dir


def create_path_mock(builtin_dist_dir):
    """Create a properly mocked Path object that returns builtin_dist_dir for the distributions path."""
    mock_parent_parent_parent = MagicMock()
    mock_parent_parent_parent.__truediv__ = lambda self, other: builtin_dist_dir if other == "distributions" else MagicMock()

    mock_path = MagicMock()
    mock_path.parent.parent.parent = mock_parent_parent_parent

    return mock_path


class TestStackList:
    """Test suite for llama stack list command."""

    def test_builtin_distros_shown_without_running(self, list_stacks_command, mock_distro_dir, tmp_path):
        """Test that built-in distributions are shown even before running them.

        This verifies the fix for issue #3922 where `llama stack list` only showed
        distributions after they were run.
        """
        mock_path = create_path_mock(mock_distro_dir)

        # Mock DISTRIBS_BASE_DIR to be a non-existent directory (no built distributions)
        with patch("llama_stack.cli.stack.list_stacks.DISTRIBS_BASE_DIR", tmp_path / "nonexistent"):
            with patch("llama_stack.cli.stack.list_stacks.Path") as mock_path_class:
                mock_path_class.return_value = mock_path

                distributions = list_stacks_command._get_distribution_dirs()

                # Verify built-in distributions are found
                assert len(distributions) > 0, "Should find built-in distributions"
                assert all(source_type == "built-in" for _, source_type in distributions.values()), "All should be built-in"

                # Check specific distributions we created
                assert "starter" in distributions
                assert "nvidia" in distributions
                assert "dell" in distributions

    def test_builtin_and_built_distros_shown_together(self, list_stacks_command, mock_distro_dir, mock_distribs_base_dir):
        """Test that both built-in and built distributions are shown together."""
        mock_path = create_path_mock(mock_distro_dir)

        with patch("llama_stack.cli.stack.list_stacks.DISTRIBS_BASE_DIR", mock_distribs_base_dir):
            with patch("llama_stack.cli.stack.list_stacks.Path") as mock_path_class:
                mock_path_class.return_value = mock_path

                distributions = list_stacks_command._get_distribution_dirs()

                # Should have built-in distributions
                builtin_count = sum(1 for _, source_type in distributions.values() if source_type == "built-in")
                # Should have built distributions
                built_count = sum(1 for _, source_type in distributions.values() if source_type == "built")

                assert builtin_count > 0, "Should have built-in distributions"
                assert built_count > 0, "Should have built distributions"

    def test_built_distribution_overrides_builtin(self, list_stacks_command, mock_distro_dir, mock_distribs_base_dir):
        """Test that built distributions override built-in ones with the same name."""
        mock_path = create_path_mock(mock_distro_dir)

        with patch("llama_stack.cli.stack.list_stacks.DISTRIBS_BASE_DIR", mock_distribs_base_dir):
            with patch("llama_stack.cli.stack.list_stacks.Path") as mock_path_class:
                mock_path_class.return_value = mock_path

                distributions = list_stacks_command._get_distribution_dirs()

                # "starter" should exist and be marked as "built" (not "built-in")
                # because the built version overrides the built-in one
                assert "starter" in distributions
                _, source_type = distributions["starter"]
                assert source_type == "built", "Built distribution should override built-in"

    def test_empty_distributions(self, list_stacks_command, tmp_path):
        """Test behavior when no distributions exist."""
        nonexistent = tmp_path / "nonexistent"
        mock_path = create_path_mock(nonexistent)

        with patch("llama_stack.cli.stack.list_stacks.DISTRIBS_BASE_DIR", nonexistent):
            with patch("llama_stack.cli.stack.list_stacks.Path") as mock_path_class:
                mock_path_class.return_value = mock_path

                distributions = list_stacks_command._get_distribution_dirs()

                assert len(distributions) == 0, "Should return empty dict when no distributions exist"

    def test_config_files_detection_builtin(self, list_stacks_command, mock_distro_dir, tmp_path):
        """Test that config files are correctly detected for built-in distributions."""
        mock_path = create_path_mock(mock_distro_dir)

        with patch("llama_stack.cli.stack.list_stacks.DISTRIBS_BASE_DIR", tmp_path / "nonexistent"):
            with patch("llama_stack.cli.stack.list_stacks.Path") as mock_path_class:
                mock_path_class.return_value = mock_path

                distributions = list_stacks_command._get_distribution_dirs()

                # Check that starter has both config files
                if "starter" in distributions:
                    path, source_type = distributions["starter"]
                    if source_type == "built-in":
                        assert (path / "build.yaml").exists()
                        assert (path / "run.yaml").exists()

    def test_config_files_detection_built(self, list_stacks_command, tmp_path):
        """Test that config files are correctly detected for built distributions."""
        # Create a built distribution
        built_dir = tmp_path / "distributions"
        built_dir.mkdir(parents=True)
        test_distro = built_dir / "test-distro"
        test_distro.mkdir()
        (test_distro / "test-distro-build.yaml").write_text("# build")
        (test_distro / "test-distro-run.yaml").write_text("# run")

        nonexistent = tmp_path / "nonexistent"
        mock_path = create_path_mock(nonexistent)

        with patch("llama_stack.cli.stack.list_stacks.DISTRIBS_BASE_DIR", built_dir):
            with patch("llama_stack.cli.stack.list_stacks.Path") as mock_path_class:
                mock_path_class.return_value = mock_path

                distributions = list_stacks_command._get_distribution_dirs()

                assert "test-distro" in distributions
                path, source_type = distributions["test-distro"]
                assert source_type == "built"
                assert (path / "test-distro-build.yaml").exists()
                assert (path / "test-distro-run.yaml").exists()

    def test_llamastack_prefix_stripped(self, list_stacks_command, tmp_path):
        """Test that llamastack- prefix is stripped from built distribution names."""
        # Create a built distribution with llamastack- prefix
        built_dir = tmp_path / "distributions"
        built_dir.mkdir(parents=True)
        distro_with_prefix = built_dir / "llamastack-mystack"
        distro_with_prefix.mkdir()

        nonexistent = tmp_path / "nonexistent"
        mock_path = create_path_mock(nonexistent)

        with patch("llama_stack.cli.stack.list_stacks.DISTRIBS_BASE_DIR", built_dir):
            with patch("llama_stack.cli.stack.list_stacks.Path") as mock_path_class:
                mock_path_class.return_value = mock_path

                distributions = list_stacks_command._get_distribution_dirs()

                # Should be listed as "mystack" not "llamastack-mystack"
                assert "mystack" in distributions
                assert "llamastack-mystack" not in distributions

    def test_hidden_directories_ignored(self, list_stacks_command, mock_distro_dir, tmp_path):
        """Test that hidden directories (starting with .) are ignored."""
        # Add a hidden directory
        hidden_dir = mock_distro_dir / ".hidden"
        hidden_dir.mkdir()
        (hidden_dir / "build.yaml").write_text("# build")

        # Add a __pycache__ directory
        pycache_dir = mock_distro_dir / "__pycache__"
        pycache_dir.mkdir()

        mock_path = create_path_mock(mock_distro_dir)

        with patch("llama_stack.cli.stack.list_stacks.DISTRIBS_BASE_DIR", tmp_path / "nonexistent"):
            with patch("llama_stack.cli.stack.list_stacks.Path") as mock_path_class:
                mock_path_class.return_value = mock_path

                distributions = list_stacks_command._get_distribution_dirs()

                assert ".hidden" not in distributions
                assert "__pycache__" not in distributions

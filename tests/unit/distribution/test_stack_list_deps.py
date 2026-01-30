# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
from io import StringIO
from unittest.mock import patch

from llama_stack.cli.stack._list_deps import (
    format_output_deps_only,
    run_stack_list_deps_command,
)


def test_stack_list_deps_basic():
    args = argparse.Namespace(
        config=None,
        env_name="test-env",
        providers="inference=remote::ollama",
        format="deps-only",
    )

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        run_stack_list_deps_command(args)
        output = mock_stdout.getvalue()

        # deps-only format should NOT include "uv pip install" or "Dependencies for"
        assert "uv pip install" not in output
        assert "Dependencies for" not in output

        # Check that expected dependencies are present
        assert "ollama" in output
        assert "aiohttp" in output
        assert "fastapi" in output


def test_stack_list_deps_with_distro_uv():
    args = argparse.Namespace(
        config="starter",
        env_name=None,
        providers=None,
        format="uv",
    )

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        run_stack_list_deps_command(args)
        output = mock_stdout.getvalue()

        assert "uv pip install" in output


def test_list_deps_formatting_quotes_only_for_uv():
    deps_only = format_output_deps_only(["mcp>=1.23.0"], [], [], uv=False)
    assert deps_only.strip() == "mcp>=1.23.0"

    uv_format = format_output_deps_only(["mcp>=1.23.0"], [], [], uv=True)
    assert uv_format.strip() == "uv pip install 'mcp>=1.23.0'"


def test_stack_list_deps_expands_provider_dependencies():
    """Test that listing deps for a provider also includes deps from its API dependencies.

    For example, agents=inline::meta-reference depends on the inference API.
    When we list deps for agents, we should also get dependencies from an inference provider.
    This test verifies the expansion happens by checking that dependencies unique to
    inference providers appear in the agents output.
    """
    # First, get dependencies for just the inference provider
    inference_args = argparse.Namespace(
        config=None,
        env_name="test-env",
        providers="inference=inline::meta-reference",
        format="deps-only",
    )

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        run_stack_list_deps_command(inference_args)
        inference_output = mock_stdout.getvalue()

    # Now get dependencies for agents, which should include inference deps
    agents_args = argparse.Namespace(
        config=None,
        env_name="test-env",
        providers="agents=inline::meta-reference",
        format="deps-only",
    )

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        run_stack_list_deps_command(agents_args)
        agents_output = mock_stdout.getvalue()

    # Verify that inference-specific dependencies appear in agents output
    # (because agents depends on inference API and dependencies were expanded)
    # Pick a few packages that are specific to inference providers
    inference_specific_packages = ["torch", "transformers", "accelerate"]

    for package in inference_specific_packages:
        assert package in inference_output, f"{package} should be in inference deps"
        assert package in agents_output, f"{package} should be in agents deps (expanded from inference dependency)"

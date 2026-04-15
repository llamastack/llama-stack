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

    For example, responses=inline::builtin depends on the inference API.
    When we list deps for responses, we should also get dependencies from an inference provider.
    This test picks a known dependency (inference), lists its deps, then verifies those
    deps appear in the responses output (proving expansion happened).
    """
    # First, get dependencies for a vector_io provider (which responses depends on)
    vector_io_args = argparse.Namespace(
        config=None,
        env_name="test-env",
        providers="vector_io=inline::faiss",
        format="deps-only",
    )

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        run_stack_list_deps_command(vector_io_args)
        vector_io_output = mock_stdout.getvalue()

    # Now get dependencies for responses, which should include vector_io deps
    responses_args = argparse.Namespace(
        config=None,
        env_name="test-env",
        providers="responses=inline::builtin",
        format="deps-only",
    )

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        run_stack_list_deps_command(responses_args)
        responses_output = mock_stdout.getvalue()

    # Verify that dependencies were expanded: responses output should include
    # vector_io-specific dependencies. Extract package names from the vector_io output
    # and verify at least some appear in the responses output.
    vector_io_deps = set(vector_io_output.split())
    responses_deps = set(responses_output.split())

    # The vector_io provider should have some dependencies
    assert len(vector_io_deps) > 0, "Vector IO provider should have dependencies"

    # At least one vector_io dependency should appear in responses output
    # (proving that dependency expansion happened)
    common_deps = vector_io_deps & responses_deps
    assert len(common_deps) > 0, (
        "Responses dependencies should include at least some vector_io dependencies, "
        "proving that dependency expansion happened"
    )

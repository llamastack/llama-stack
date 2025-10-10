# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import json
from io import StringIO
from unittest.mock import patch

from llama_stack.cli.stack._show import (
    run_stack_show_command,
)


def test_stack_show_basic():
    args = argparse.Namespace(
        config=None,
        distro=None,
        env_name="test-env",
        providers="inference=remote::ollama",
        format="plain",
    )

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        run_stack_show_command(args)
        output = mock_stdout.getvalue()

        assert "# Dependencies for test-env" in output
        assert "uv pip install" in output


def test_stack_show_with_distro():
    args = argparse.Namespace(
        config=None,
        distro="starter",
        env_name=None,
        providers=None,
        format="plain",
    )

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        run_stack_show_command(args)
        output = mock_stdout.getvalue()

        assert "# Dependencies for starter" in output
        assert "uv pip install" in output


def test_stack_show_json_format():
    args = argparse.Namespace(
        config=None,
        distro="starter",
        env_name=None,
        providers=None,
        format="json",
    )

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        run_stack_show_command(args)
        output = mock_stdout.getvalue()

        # Parse JSON to verify structure
        data = json.loads(output)
        assert "name" in data
        assert data["name"] == "starter"
        assert "apis" in data
        assert isinstance(data["apis"], list)
        assert "pip_dependencies" in data
        assert isinstance(data["pip_dependencies"], list)
        assert len(data["pip_dependencies"]) > 0
        assert "special_pip_dependencies" in data
        assert isinstance(data["special_pip_dependencies"], list)


def test_stack_show_json_format_with_providers():
    args = argparse.Namespace(
        config=None,
        distro=None,
        env_name="test-env",
        providers="inference=remote::ollama",
        format="json",
    )

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        run_stack_show_command(args)
        output = mock_stdout.getvalue()

        # Parse JSON to verify structure
        data = json.loads(output)
        assert data["name"] == "test-env"
        assert len(data["apis"]) > 0
        assert any(api["api"] == "inference" and api["provider"] == "remote::ollama" for api in data["apis"])

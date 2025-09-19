# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
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
    )

    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        run_stack_show_command(args)
        output = mock_stdout.getvalue()

        assert "# Dependencies for starter" in output
        assert "uv pip install" in output

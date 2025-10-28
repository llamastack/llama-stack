# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Backward compatibility test for run.yaml files.

This test ensures that changes to StackRunConfig don't break
existing run.yaml files from previous versions.
"""

import os
from pathlib import Path

import pytest
import yaml

from llama_stack.core.datatypes import StackRunConfig


def get_test_configs():
    configs_dir = os.environ.get("COMPAT_TEST_CONFIGS_DIR")
    if configs_dir:
        # CI mode: test configs extracted from main/release
        config_dir = Path(configs_dir)
        if not config_dir.exists():
            pytest.skip(f"Config directory not found: {configs_dir}")

        config_files = sorted(config_dir.glob("*.yaml"))
        if not config_files:
            pytest.skip(f"No .yaml files found in {configs_dir}")

        return config_files
    else:
        # Local mode: test current distribution configs
        repo_root = Path(__file__).parent.parent.parent
        config_files = sorted((repo_root / "src" / "llama_stack" / "distributions").glob("*/run.yaml"))

        if not config_files:
            pytest.skip("No run.yaml files found in distributions/")

        return config_files


@pytest.mark.parametrize("config_file", get_test_configs(), ids=lambda p: p.stem)
def test_load_run_config(config_file):
    with open(config_file) as f:
        config_data = yaml.safe_load(f)

    StackRunConfig.model_validate(config_data)

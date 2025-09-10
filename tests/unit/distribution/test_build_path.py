# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path

from llama_stack.cli.stack._build import (
    _run_stack_build_command_from_build_config,
)
from llama_stack.core.datatypes import BuildConfig, DistributionSpec
from llama_stack.core.utils.image_types import LlamaStackImageType


def test_container_build_passes_path(monkeypatch, tmp_path):
    called_with = {}

    def spy_build_image(build_config, image_name, distro_or_config, run_config=None):
        called_with["path"] = distro_or_config
        called_with["run_config"] = run_config
        return 0

    monkeypatch.setattr(
        "llama_stack.cli.stack._build.build_image",
        spy_build_image,
        raising=True,
    )

    cfg = BuildConfig(
        image_type=LlamaStackImageType.CONTAINER.value,
        distribution_spec=DistributionSpec(providers={}, description=""),
    )

    _run_stack_build_command_from_build_config(cfg, image_name="dummy")

    assert "path" in called_with
    assert isinstance(called_with["path"], str)
    assert Path(called_with["path"]).exists()
    assert called_with["run_config"] is None

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from llama_stack.cli.stack._build import _apply_single_provider_filter
from llama_stack.core.datatypes import BuildConfig, BuildProvider, DistributionSpec
from llama_stack.core.utils.image_types import LlamaStackImageType


def test_filters_single_api():
    """Test filtering keeps only specified provider for one API."""
    build_config = BuildConfig(
        image_type=LlamaStackImageType.VENV.value,
        distribution_spec=DistributionSpec(
            providers={
                "vector_io": [
                    BuildProvider(provider_type="inline::faiss"),
                    BuildProvider(provider_type="inline::sqlite-vec"),
                ],
                "inference": [
                    BuildProvider(provider_type="remote::openai"),
                ],
            },
            description="Test",
        ),
    )

    filtered = _apply_single_provider_filter(build_config, "vector_io=inline::sqlite-vec")

    assert len(filtered.distribution_spec.providers["vector_io"]) == 1
    assert filtered.distribution_spec.providers["vector_io"][0].provider_type == "inline::sqlite-vec"
    assert len(filtered.distribution_spec.providers["inference"]) == 1  # unchanged


def test_filters_multiple_apis():
    """Test filtering multiple APIs."""
    build_config = BuildConfig(
        image_type=LlamaStackImageType.VENV.value,
        distribution_spec=DistributionSpec(
            providers={
                "vector_io": [
                    BuildProvider(provider_type="inline::faiss"),
                    BuildProvider(provider_type="inline::sqlite-vec"),
                ],
                "inference": [
                    BuildProvider(provider_type="remote::openai"),
                    BuildProvider(provider_type="remote::anthropic"),
                ],
            },
            description="Test",
        ),
    )

    filtered = _apply_single_provider_filter(build_config, "vector_io=inline::faiss,inference=remote::openai")

    assert len(filtered.distribution_spec.providers["vector_io"]) == 1
    assert filtered.distribution_spec.providers["vector_io"][0].provider_type == "inline::faiss"
    assert len(filtered.distribution_spec.providers["inference"]) == 1
    assert filtered.distribution_spec.providers["inference"][0].provider_type == "remote::openai"


def test_provider_not_found_exits():
    """Test error when specified provider doesn't exist."""
    build_config = BuildConfig(
        image_type=LlamaStackImageType.VENV.value,
        distribution_spec=DistributionSpec(
            providers={"vector_io": [BuildProvider(provider_type="inline::faiss")]},
            description="Test",
        ),
    )

    with pytest.raises(SystemExit):
        _apply_single_provider_filter(build_config, "vector_io=inline::nonexistent")


def test_invalid_format_exits():
    """Test error for invalid filter format."""
    build_config = BuildConfig(
        image_type=LlamaStackImageType.VENV.value,
        distribution_spec=DistributionSpec(providers={}, description="Test"),
    )

    with pytest.raises(SystemExit):
        _apply_single_provider_filter(build_config, "invalid_format")

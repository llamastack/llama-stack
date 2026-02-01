# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.cli.stack.utils import add_dependent_providers
from llama_stack.core.datatypes import Provider
from llama_stack.core.distribution import get_provider_registry


def test_add_dependent_providers_expands_required_apis():
    provider_registry = get_provider_registry()
    provider_list = {
        "agents": [
            Provider(
                provider_type="inline::meta-reference",
                provider_id="meta-reference",
            )
        ]
    }

    add_dependent_providers(
        provider_list=provider_list,
        provider_registry=provider_registry,
        requested_provider_types=["inline::meta-reference"],
    )

    # Required API dependencies for agents should be present.
    assert "inference" in provider_list
    assert "vector_io" in provider_list
    assert "tool_runtime" in provider_list
    assert "files" in provider_list

    # Providers should be added for those APIs.
    assert provider_list["inference"]
    assert provider_list["vector_io"]
    assert provider_list["tool_runtime"]
    assert provider_list["files"]


def test_add_dependent_providers_include_configs():
    provider_registry = get_provider_registry()
    provider_list = {
        "agents": [
            Provider(
                provider_type="inline::meta-reference",
                provider_id="meta-reference",
            )
        ]
    }

    add_dependent_providers(
        provider_list=provider_list,
        provider_registry=provider_registry,
        requested_provider_types=["inline::meta-reference"],
        include_configs=True,
        distro_dir="~/.llama/distributions/providers-run",
    )

    inference_provider = provider_list["inference"][0]
    assert inference_provider.config, "Expected sample config for inference provider"

    files_provider = provider_list["files"][0]
    assert "storage_dir" in files_provider.config

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_api import Api, InlineProviderSpec


def get_provider_spec() -> InlineProviderSpec:
    return InlineProviderSpec(
        api=Api.tool_runtime,
        provider_type="inline::file-search",
        # Dependencies are managed by this package's pyproject.toml and
        # installed automatically via the uv workspace.
        pip_packages=[],
        module="llama_stack_provider_tool_runtime_file_search",
        config_class="llama_stack_provider_tool_runtime_file_search.config.FileSearchToolRuntimeConfig",
        api_dependencies=[Api.vector_io, Api.inference, Api.files],
        toolgroup_id="builtin::file_search",
        description="File search tool runtime for document ingestion, chunking, and semantic search.",
    )

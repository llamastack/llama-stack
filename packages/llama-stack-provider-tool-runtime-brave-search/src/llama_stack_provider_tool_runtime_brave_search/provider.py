# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack_api import Api, RemoteProviderSpec


def get_provider_spec() -> RemoteProviderSpec:
    return RemoteProviderSpec(
        api=Api.tool_runtime,
        provider_type="remote::brave-search",
        adapter_type="brave-search",
        pip_packages=[],
        module="llama_stack_provider_tool_runtime_brave_search",
        config_class="llama_stack_provider_tool_runtime_brave_search.config.BraveSearchToolConfig",
        provider_data_validator="llama_stack_provider_tool_runtime_brave_search.BraveSearchToolProviderDataValidator",
        toolgroup_id="builtin::websearch",
        description="Brave Search tool for web search capabilities with privacy-focused results.",
    )

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack_api import Api, RemoteProviderSpec


def get_provider_spec() -> RemoteProviderSpec:
    return RemoteProviderSpec(
        api=Api.tool_runtime,
        provider_type="remote::bing-search",
        adapter_type="bing-search",
        pip_packages=[],
        module="llama_stack_provider_tool_runtime_bing_search",
        config_class="llama_stack_provider_tool_runtime_bing_search.config.BingSearchToolConfig",
        provider_data_validator="llama_stack_provider_tool_runtime_bing_search.BingSearchToolProviderDataValidator",
        toolgroup_id="builtin::websearch",
        description="Bing Search tool for web search capabilities using Microsoft's search engine.",
    )

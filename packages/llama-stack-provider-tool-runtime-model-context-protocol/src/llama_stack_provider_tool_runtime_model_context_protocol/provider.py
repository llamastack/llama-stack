# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack_api import Api, RemoteProviderSpec


def get_provider_spec() -> RemoteProviderSpec:
    return RemoteProviderSpec(
        api=Api.tool_runtime,
        provider_type="remote::model-context-protocol",
        adapter_type="model-context-protocol",
        pip_packages=[],
        module="llama_stack_provider_tool_runtime_model_context_protocol",
        config_class="llama_stack_provider_tool_runtime_model_context_protocol.config.MCPProviderConfig",
        provider_data_validator="llama_stack_provider_tool_runtime_model_context_protocol.config.MCPProviderDataValidator",
        description="Model Context Protocol (MCP) tool for standardized tool calling and context management.",
    )

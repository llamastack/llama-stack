# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack_api import Api, RemoteProviderSpec


def get_provider_spec() -> RemoteProviderSpec:
    return RemoteProviderSpec(
        api=Api.tool_runtime,
        provider_type="remote::wolfram-alpha",
        adapter_type="wolfram-alpha",
        pip_packages=[],
        module="llama_stack_provider_tool_runtime_wolfram_alpha",
        config_class="llama_stack_provider_tool_runtime_wolfram_alpha.config.WolframAlphaToolConfig",
        provider_data_validator="llama_stack_provider_tool_runtime_wolfram_alpha.WolframAlphaToolProviderDataValidator",
        toolgroup_id="builtin::wolfram_alpha",
        description="Wolfram Alpha tool for computational knowledge and mathematical calculations.",
    )

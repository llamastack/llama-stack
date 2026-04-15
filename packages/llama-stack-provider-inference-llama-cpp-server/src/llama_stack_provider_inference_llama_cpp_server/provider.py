# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_api import Api, RemoteProviderSpec


def get_provider_spec() -> RemoteProviderSpec:
    return RemoteProviderSpec(
        api=Api.inference,
        adapter_type="llama-cpp-server",
        provider_type="remote::llama-cpp-server",
        pip_packages=[],
        module="llama_stack_provider_inference_llama_cpp_server",
        config_class="llama_stack_provider_inference_llama_cpp_server.config.LlamaCppServerConfig",
        description="llama.cpp inference provider for connecting to llama.cpp servers with OpenAI-compatible API.",
    )

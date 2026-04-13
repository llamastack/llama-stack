# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_api import Api, RemoteProviderSpec


def get_provider_spec() -> RemoteProviderSpec:
    return RemoteProviderSpec(
        api=Api.inference,
        adapter_type="ollama",
        provider_type="remote::ollama",
        # Dependencies are managed by this package's pyproject.toml and
        # installed automatically via the uv workspace.
        pip_packages=[],
        config_class="llama_stack_provider_inference_ollama.config.OllamaImplConfig",
        module="llama_stack_provider_inference_ollama",
        description="Ollama inference provider for running local models through the Ollama runtime.",
    )

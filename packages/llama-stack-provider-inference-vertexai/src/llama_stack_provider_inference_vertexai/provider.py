# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_api import Api, RemoteProviderSpec


def get_provider_spec() -> RemoteProviderSpec:
    return RemoteProviderSpec(
        api=Api.inference,
        adapter_type="vertexai",
        provider_type="remote::vertexai",
        pip_packages=[],
        module="llama_stack_provider_inference_vertexai",
        config_class="llama_stack_provider_inference_vertexai.VertexAIConfig",
        provider_data_validator="llama_stack_provider_inference_vertexai.config.VertexAIProviderDataValidator",
        description="Google Vertex AI inference provider for accessing Gemini models through Google Cloud's Vertex AI platform.",
    )

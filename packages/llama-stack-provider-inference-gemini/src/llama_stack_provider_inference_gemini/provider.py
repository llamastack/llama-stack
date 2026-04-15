# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_api import Api, RemoteProviderSpec


def get_provider_spec() -> RemoteProviderSpec:
    return RemoteProviderSpec(
        api=Api.inference,
        adapter_type="gemini",
        provider_type="remote::gemini",
        pip_packages=[],
        module="llama_stack_provider_inference_gemini",
        config_class="llama_stack_provider_inference_gemini.GeminiConfig",
        provider_data_validator="llama_stack_provider_inference_gemini.config.GeminiProviderDataValidator",
        description="Google Gemini inference provider for accessing Gemini models and Google's AI services.",
    )

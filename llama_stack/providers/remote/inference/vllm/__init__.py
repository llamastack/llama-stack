# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel, Field, SecretStr

from .config import VLLMInferenceAdapterConfig


class VLLMProviderDataValidator(BaseModel):
    vllm_api_token: SecretStr = Field(
        description="API token for vLLM models",
    )


async def get_adapter_impl(config: VLLMInferenceAdapterConfig, _deps):
    from .vllm import VLLMInferenceAdapter

    assert isinstance(config, VLLMInferenceAdapterConfig), f"Unexpected config type: {type(config)}"
    impl = VLLMInferenceAdapter(config)
    await impl.initialize()
    return impl

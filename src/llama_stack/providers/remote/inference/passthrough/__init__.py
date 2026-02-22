# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel, SecretStr

from .config import PassthroughImplConfig


class PassthroughProviderDataValidator(BaseModel):
    passthrough_url: str
    passthrough_api_key: SecretStr


async def get_adapter_impl(config: PassthroughImplConfig, _deps):
    from .passthrough import PassthroughInferenceAdapter

    assert isinstance(config, PassthroughImplConfig), f"Unexpected config type: {type(config)}"
    impl = PassthroughInferenceAdapter(config)
    await impl.initialize()
    return impl

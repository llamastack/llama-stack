# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel

from .config import TogetherImplConfig


class TogetherProviderDataValidator(BaseModel):
    together_api_key: str


async def get_adapter_impl(config: TogetherImplConfig, _deps):
    from .together import TogetherInferenceAdapter

    assert isinstance(config, TogetherImplConfig), f"Unexpected config type: {type(config)}"
    impl = TogetherInferenceAdapter(config=config)
    await impl.initialize()
    return impl

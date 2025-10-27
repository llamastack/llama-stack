# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel

from .brave_search import BraveSearchToolRuntimeImpl
from .config import BraveSearchToolConfig


class BraveSearchToolProviderDataValidator(BaseModel):
    brave_search_api_key: str


async def get_adapter_impl(config: BraveSearchToolConfig, _deps):
    impl = BraveSearchToolRuntimeImpl(config)
    await impl.initialize()
    return impl

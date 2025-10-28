# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel

from .config import TavilySearchToolConfig
from .tavily_search import TavilySearchToolRuntimeImpl


class TavilySearchToolProviderDataValidator(BaseModel):
    tavily_search_api_key: str


async def get_adapter_impl(config: TavilySearchToolConfig, _deps):
    impl = TavilySearchToolRuntimeImpl(config)
    await impl.initialize()
    return impl

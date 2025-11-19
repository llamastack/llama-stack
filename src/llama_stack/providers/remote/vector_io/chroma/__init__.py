# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.core.datatypes import VectorStoresConfig
from llama_stack_api import Api, ProviderSpec

from .config import ChromaVectorIOConfig


async def get_adapter_impl(
    config: ChromaVectorIOConfig, deps: dict[Api, ProviderSpec], vector_stores_config: VectorStoresConfig | None = None
):
    from .chroma import ChromaVectorIOAdapter

    impl = ChromaVectorIOAdapter(config, deps[Api.inference], deps.get(Api.files), vector_stores_config)
    await impl.initialize()
    return impl

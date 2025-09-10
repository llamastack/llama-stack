# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.providers.datatypes import Api, ProviderSpec

from .config import QdrantVectorIOConfig


async def get_adapter_impl(config: QdrantVectorIOConfig, deps: dict[Api, ProviderSpec]):
    from .qdrant import QdrantVectorIOAdapter

    files_api = deps.get(Api.files)
    impl = QdrantVectorIOAdapter(config, deps[Api.inference], files_api)
    await impl.initialize()
    return impl

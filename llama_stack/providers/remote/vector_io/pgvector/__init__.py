# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.providers.datatypes import Api, ProviderSpec

from .config import PGVectorVectorIOConfig


async def get_adapter_impl(config: PGVectorVectorIOConfig, deps: dict[Api, ProviderSpec]):
    from .pgvector import PGVectorVectorIOAdapter

    impl = PGVectorVectorIOAdapter(config, deps[Api.inference], deps.get(Api.files))
    await impl.initialize()
    return impl

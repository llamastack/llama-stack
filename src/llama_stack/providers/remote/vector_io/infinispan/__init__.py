# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_api import Api, ProviderSpec

from .config import InfinispanVectorIOConfig


async def get_adapter_impl(config: InfinispanVectorIOConfig, deps: dict[Api, ProviderSpec]):
    from .infinispan import InfinispanVectorIOAdapter

    impl = InfinispanVectorIOAdapter(config, deps[Api.inference], deps.get(Api.files))
    await impl.initialize()
    return impl

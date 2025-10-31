# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.providers.datatypes import Api, ProviderSpec

from .config import MongoDBVectorIOConfig


async def get_adapter_impl(config: MongoDBVectorIOConfig, deps: dict[Api, ProviderSpec]):
    from .mongodb import MongoDBVectorIOAdapter

    # Handle the deps resolution - if files API exists, pass it, otherwise None
    files_api = deps.get(Api.files)
    models_api = deps.get(Api.models)
    impl = MongoDBVectorIOAdapter(config, deps[Api.inference], files_api, models_api)
    await impl.initialize()
    return impl

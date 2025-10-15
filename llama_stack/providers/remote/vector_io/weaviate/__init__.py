# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.core.datatypes import StackRunConfig
from llama_stack.providers.datatypes import Api, ProviderSpec

from .config import WeaviateVectorIOConfig


async def get_adapter_impl(
    config: WeaviateVectorIOConfig, deps: dict[Api, ProviderSpec], run_config: StackRunConfig | None = None
):
    from .weaviate import WeaviateVectorIOAdapter

    vector_stores_config = None
    if run_config and run_config.vector_stores:
        vector_stores_config = run_config.vector_stores

    impl = WeaviateVectorIOAdapter(
        config,
        deps[Api.inference],
        deps[Api.models],
        deps.get(Api.files),
        vector_stores_config,
    )
    await impl.initialize()
    return impl

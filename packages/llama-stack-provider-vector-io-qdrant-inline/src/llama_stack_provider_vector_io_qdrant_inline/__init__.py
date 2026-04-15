# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack_provider_vector_io_qdrant.inline_config import QdrantVectorIOConfig

from llama_stack_api import Api


async def get_provider_impl(config: QdrantVectorIOConfig, deps: dict[Api, Any]):
    from llama_stack_provider_vector_io_qdrant.qdrant import QdrantVectorIOAdapter

    impl = QdrantVectorIOAdapter(config, deps[Api.inference], deps.get(Api.files), deps.get(Api.file_processors))
    await impl.initialize()
    return impl

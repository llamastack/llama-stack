# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.providers.datatypes import Api

from .config import QdrantVectorIOConfig


async def get_provider_impl(config: QdrantVectorIOConfig, deps: dict[Api, Any]):
    from llama_stack.providers.remote.vector_io.qdrant.qdrant import QdrantVectorIOAdapter

    assert isinstance(config, QdrantVectorIOConfig), f"Unexpected config type: {type(config)}"
    files_api = deps.get(Api.files)
    impl = QdrantVectorIOAdapter(config, deps[Api.inference], files_api)
    await impl.initialize()
    return impl

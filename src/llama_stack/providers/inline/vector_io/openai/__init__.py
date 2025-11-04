# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.providers.datatypes import Api

from .config import OpenAIVectorIOConfig


async def get_provider_impl(config: OpenAIVectorIOConfig, deps: dict[Api, Any]):
    from .openai import OpenAIVectorIOAdapter

    assert isinstance(config, OpenAIVectorIOConfig), f"Unexpected config type: {type(config)}"

    impl = OpenAIVectorIOAdapter(
        config,
        deps[Api.inference],
        deps.get(Api.files),
    )
    await impl.initialize()
    return impl

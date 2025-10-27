# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import HuggingfaceDatasetIOConfig


async def get_adapter_impl(
    config: HuggingfaceDatasetIOConfig,
    _deps,
):
    from .huggingface import HuggingfaceDatasetIOImpl

    impl = HuggingfaceDatasetIOImpl(config)
    await impl.initialize()
    return impl

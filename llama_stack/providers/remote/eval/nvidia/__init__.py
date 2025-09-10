# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Any

from llama_stack.core.datatypes import Api

from .config import NVIDIAEvalConfig


async def get_adapter_impl(
    config: NVIDIAEvalConfig,
    deps: dict[Api, Any],
):
    from .eval import NVIDIAEvalImpl

    impl = NVIDIAEvalImpl(
        config,
        deps[Api.datasetio],
        deps[Api.datasets],
        deps[Api.scoring],
        deps[Api.inference],
        deps[Api.agents],
    )
    await impl.initialize()
    return impl


__all__ = ["get_adapter_impl", "NVIDIAEvalImpl"]

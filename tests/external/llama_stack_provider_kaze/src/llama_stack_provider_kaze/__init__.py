# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Kaze weather provider for Llama Stack."""

from .config import KazeProviderConfig
from .kaze import WeatherKazeAdapter

__all__ = ["KazeProviderConfig", "WeatherKazeAdapter"]


async def get_adapter_impl(config: KazeProviderConfig, _deps):
    from .kaze import WeatherKazeAdapter

    impl = WeatherKazeAdapter(config)
    await impl.initialize()
    return impl

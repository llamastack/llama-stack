# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Protocol

from llama_stack.providers.datatypes import AdapterSpec, Api, ProviderSpec, RemoteProviderSpec
from llama_stack.schema_utils import webmethod


def available_providers() -> list[ProviderSpec]:
    return [
        RemoteProviderSpec(
            api=Api.weather,
            provider_type="remote::kaze",
            config_class="llama_stack_provider_kaze.KazeProviderConfig",
            adapter=AdapterSpec(
                adapter_type="kaze",
                module="llama_stack_provider_kaze",
                pip_packages=["llama_stack_provider_kaze"],
                config_class="llama_stack_provider_kaze.KazeProviderConfig",
            ),
        ),
    ]


class WeatherProvider(Protocol):
    """
    A protocol for the Weather API.
    """

    @webmethod(route="/weather/locations", method="GET")
    async def get_available_locations() -> dict[str, list[str]]:
        """
        Get the available locations.
        """
        ...

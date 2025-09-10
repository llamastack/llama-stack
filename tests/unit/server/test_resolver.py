# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import inspect
import sys
from typing import Any, Protocol
from unittest.mock import AsyncMock, MagicMock

from pydantic import BaseModel, Field

from llama_stack.apis.inference import Inference
from llama_stack.core.datatypes import (
    Api,
    Provider,
    StackRunConfig,
)
from llama_stack.core.resolver import resolve_impls
from llama_stack.core.routers.inference import InferenceRouter
from llama_stack.core.routing_tables.models import ModelsRoutingTable
from llama_stack.providers.datatypes import InlineProviderSpec, ProviderSpec


def add_protocol_methods(cls: type, protocol: type[Protocol]) -> None:
    """Dynamically add protocol methods to a class by inspecting the protocol."""
    for name, value in inspect.getmembers(protocol):
        if inspect.isfunction(value) and hasattr(value, "__webmethod__"):
            # Get the signature
            sig = inspect.signature(value)

            # Create an async function with the same signature that returns a MagicMock
            async def mock_impl(*args, **kwargs):
                return MagicMock()

            # Set the signature on our mock implementation
            mock_impl.__signature__ = sig
            # Add it to the class
            setattr(cls, name, mock_impl)


class SampleConfig(BaseModel):
    foo: str = Field(
        default="bar",
        description="foo",
    )

    @classmethod
    def sample_run_config(cls, **kwargs: Any) -> dict[str, Any]:
        return {
            "foo": "baz",
        }


class SampleImpl:
    def __init__(self, config: SampleConfig, deps: dict[Api, Any], provider_spec: ProviderSpec = None):
        self.__provider_id__ = "test_provider"
        self.__provider_spec__ = provider_spec
        self.__provider_config__ = config
        self.__deps__ = deps
        self.foo = config.foo

    async def initialize(self):
        pass


async def test_resolve_impls_basic():
    # Create a real provider spec
    provider_spec = InlineProviderSpec(
        api=Api.inference,
        provider_type="sample",
        module="test_module",
        config_class="test_resolver.SampleConfig",
        api_dependencies=[],
    )

    # Create provider registry with our provider
    provider_registry = {Api.inference: {provider_spec.provider_type: provider_spec}}

    run_config = StackRunConfig(
        image_name="test_image",
        providers={
            "inference": [
                Provider(
                    provider_id="sample_provider",
                    provider_type="sample",
                    config=SampleConfig.sample_run_config(),
                )
            ]
        },
    )

    dist_registry = MagicMock()

    mock_module = MagicMock()
    impl = SampleImpl(SampleConfig(foo="baz"), {}, provider_spec)
    add_protocol_methods(SampleImpl, Inference)

    mock_module.get_provider_impl = AsyncMock(return_value=impl)
    mock_module.get_provider_impl.__text_signature__ = "()"
    sys.modules["test_module"] = mock_module

    impls = await resolve_impls(run_config, provider_registry, dist_registry, policy={})

    assert Api.inference in impls
    assert isinstance(impls[Api.inference], InferenceRouter)

    table = impls[Api.inference].routing_table
    assert isinstance(table, ModelsRoutingTable)

    impl = table.impls_by_provider_id["sample_provider"]
    assert isinstance(impl, SampleImpl)
    assert impl.foo == "baz"
    assert impl.__provider_id__ == "sample_provider"
    assert impl.__provider_spec__ == provider_spec

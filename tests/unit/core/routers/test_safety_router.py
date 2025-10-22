# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock

from llama_stack.apis.safety.safety import ModerationObject, ModerationObjectResults
from llama_stack.apis.shields import ListShieldsResponse, Shield
from llama_stack.core.datatypes import SafetyConfig
from llama_stack.core.routers.safety import SafetyRouter


async def test_run_moderation_uses_default_shield_when_model_missing():
    routing_table = AsyncMock()
    shield = Shield(
        identifier="shield-1",
        provider_resource_id="provider/shield-model",
        provider_id="provider-id",
        params={},
    )
    routing_table.list_shields.return_value = ListShieldsResponse(data=[shield])

    moderation_response = ModerationObject(
        id="mid",
        model="shield-1",
        results=[ModerationObjectResults(flagged=False)],
    )
    provider = AsyncMock()
    provider.run_moderation.return_value = moderation_response
    routing_table.get_provider_impl.return_value = provider

    router = SafetyRouter(routing_table=routing_table, safety_config=SafetyConfig(default_shield_id="shield-1"))

    result = await router.run_moderation("hello world")

    assert result is moderation_response
    routing_table.get_provider_impl.assert_awaited_once_with("shield-1")
    provider.run_moderation.assert_awaited_once()
    _, kwargs = provider.run_moderation.call_args
    assert kwargs["model"] == "provider/shield-model"
    assert kwargs["input"] == "hello world"

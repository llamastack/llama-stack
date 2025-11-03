# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Protocol, runtime_checkable

from llama_stack.apis.inference import OpenAIMessageParam
from llama_stack.apis.shields import Shield
from llama_stack.core.telemetry.trace_protocol import trace_protocol

from .models import ModerationObject, RunShieldResponse


class ShieldStore(Protocol):
    async def get_shield(self, identifier: str) -> Shield: ...


@runtime_checkable
@trace_protocol
class SafetyService(Protocol):
    """Safety

    OpenAI-compatible Moderations API.
    """

    shield_store: ShieldStore

    async def run_shield(
        self,
        shield_id: str,
        messages: list[OpenAIMessageParam],
        params: dict[str, Any],
    ) -> RunShieldResponse:
        """Run shield."""
        ...

    async def run_moderation(self, input: str | list[str], model: str | None = None) -> ModerationObject:
        """Create moderation."""
        ...

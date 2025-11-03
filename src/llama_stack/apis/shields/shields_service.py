# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Protocol, runtime_checkable

from llama_stack.core.telemetry.trace_protocol import trace_protocol

from .models import ListShieldsResponse, Shield


@runtime_checkable
@trace_protocol
class ShieldsService(Protocol):
    async def list_shields(self) -> ListShieldsResponse:
        """List all shields."""
        ...

    async def get_shield(self, identifier: str) -> Shield:
        """Get a shield by its identifier."""
        ...

    async def register_shield(
        self,
        shield_id: str,
        provider_shield_id: str | None = None,
        provider_id: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> Shield:
        """Register a shield."""
        ...

    async def unregister_shield(self, identifier: str) -> None:
        """Unregister a shield."""
        ...

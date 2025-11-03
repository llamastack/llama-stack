# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Protocol, runtime_checkable

from .models import HealthInfo, ListRoutesResponse, VersionInfo


@runtime_checkable
class InspectService(Protocol):
    """Inspect

    APIs for inspecting the Llama Stack service, including health status, available API routes with methods and implementing providers.
    """

    async def list_routes(self) -> ListRoutesResponse:
        """List routes."""
        ...

    async def health(self) -> HealthInfo:
        """Get health status."""
        ...

    async def version(self) -> VersionInfo:
        """Get version."""
        ...

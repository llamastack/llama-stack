# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Protocol, runtime_checkable

# Import types from admin module to avoid duplication
from llama_stack_api.admin.models import (
    ApiFilter,
    HealthInfo,
    ListRoutesResponse,
    VersionInfo,
)
from llama_stack_api.schema_utils import webmethod
from llama_stack_api.version import LLAMA_STACK_API_V1


@runtime_checkable
class Inspect(Protocol):
    """Inspect

    APIs for inspecting the Llama Stack service, including health status, available API routes with methods and implementing providers.
    """

    @webmethod(route="/inspect/routes", method="GET", level=LLAMA_STACK_API_V1, deprecated=True)
    async def list_routes(self, api_filter: ApiFilter | None = None) -> ListRoutesResponse:
        """List routes.

        List all available API routes with their methods and implementing providers.

        :param api_filter: Optional filter to control which routes are returned. Can be an API level ('v1', 'v1alpha', 'v1beta') to show non-deprecated routes at that level, or 'deprecated' to show deprecated routes across all levels. If not specified, returns all non-deprecated routes.
        :returns: Response containing information about all available routes.
        """
        ...

    @webmethod(route="/health", method="GET", level=LLAMA_STACK_API_V1, require_authentication=False, deprecated=True)
    async def health(self) -> HealthInfo:
        """Get health status.

        Get the current health status of the service.

        :returns: Health information indicating if the service is operational.
        """
        ...

    @webmethod(route="/version", method="GET", level=LLAMA_STACK_API_V1, require_authentication=False, deprecated=True)
    async def version(self) -> VersionInfo:
        """Get version.

        Get the version of the service.

        :returns: Version information containing the service version number.
        """
        ...

# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Protocol, runtime_checkable

from ogx_api.safety.datatypes import ModerationObject, ShieldStore

from .models import RunModerationRequest


@runtime_checkable
class Safety(Protocol):
    """Safety API for content moderation.

    OpenAI-compatible Moderations API.
    """

    shield_store: ShieldStore

    async def run_moderation(self, request: RunModerationRequest) -> ModerationObject:
        """Classify if inputs are potentially harmful."""
        ...

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Import routes to trigger router registration
from . import routes  # noqa: F401
from .models import (
    CreatePromptRequest,
    ListPromptsResponse,
    Prompt,
    SetDefaultVersionRequest,
    UpdatePromptRequest,
)
from .prompts_service import PromptService

# Backward compatibility - export Prompts as alias for PromptService
Prompts = PromptService

__all__ = [
    "Prompts",
    "PromptService",
    "Prompt",
    "ListPromptsResponse",
    "CreatePromptRequest",
    "UpdatePromptRequest",
    "SetDefaultVersionRequest",
]

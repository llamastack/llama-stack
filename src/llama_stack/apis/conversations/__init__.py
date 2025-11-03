# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Import routes to trigger router registration
from . import routes  # noqa: F401
from .conversations_service import ConversationService
from .models import (
    Conversation,
    ConversationCreateRequest,
    ConversationDeletedResource,
    ConversationItem,
    ConversationItemCreateRequest,
    ConversationItemDeletedResource,
    ConversationItemInclude,
    ConversationItemList,
    ConversationMessage,
    ConversationUpdateRequest,
    Metadata,
)

# Backward compatibility - export Conversations as alias for ConversationService
Conversations = ConversationService

__all__ = [
    "Conversations",
    "ConversationService",
    "Conversation",
    "ConversationMessage",
    "ConversationItem",
    "ConversationCreateRequest",
    "ConversationUpdateRequest",
    "ConversationDeletedResource",
    "ConversationItemCreateRequest",
    "ConversationItemList",
    "ConversationItemDeletedResource",
    "ConversationItemInclude",
    "Metadata",
]

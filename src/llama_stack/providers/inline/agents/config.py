# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel


class AgentsConfig(BaseModel):
    """Configuration for the built-in Agents provider.

    This provider implements the Anthropic Managed Agents API for creating and managing
    agent configurations, sessions, and event-driven interactions.
    """

    # Future: Add storage backend configuration (in-memory, SQLite, PostgreSQL)
    pass

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Import routes to trigger router registration
from . import routes  # noqa: F401
from .models import (
    InvokeToolRequest,
    ListToolDefsResponse,
    ListToolGroupsResponse,
    RegisterToolGroupRequest,
    SpecialToolGroup,
    ToolDef,
    ToolGroup,
    ToolGroupInput,
    ToolInvocationResult,
)
from .rag_tool import RAGDocument, RAGQueryConfig, RAGQueryResult, RAGToolRuntime
from .tool_groups_service import ToolGroupsService, ToolStore
from .tool_runtime_service import ToolRuntimeService

# Backward compatibility - export as aliases
ToolGroups = ToolGroupsService
ToolRuntime = ToolRuntimeService

__all__ = [
    "ToolGroups",
    "ToolGroupsService",
    "ToolRuntime",
    "ToolRuntimeService",
    "ToolStore",
    "ToolDef",
    "ToolGroup",
    "ToolGroupInput",
    "ToolInvocationResult",
    "ListToolGroupsResponse",
    "ListToolDefsResponse",
    "RegisterToolGroupRequest",
    "InvokeToolRequest",
    "SpecialToolGroup",
    "RAGToolRuntime",
    "RAGDocument",
    "RAGQueryConfig",
    "RAGQueryResult",
]

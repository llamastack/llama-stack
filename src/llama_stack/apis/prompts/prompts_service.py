# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Protocol, runtime_checkable

from llama_stack.core.telemetry.trace_protocol import trace_protocol

from .models import ListPromptsResponse, Prompt


@runtime_checkable
@trace_protocol
class PromptService(Protocol):
    """Prompts

    Protocol for prompt management operations."""

    async def list_prompts(self) -> ListPromptsResponse:
        """List all prompts."""
        ...

    async def list_prompt_versions(
        self,
        prompt_id: str,
    ) -> ListPromptsResponse:
        """List prompt versions."""
        ...

    async def get_prompt(
        self,
        prompt_id: str,
        version: int | None = None,
    ) -> Prompt:
        """Get prompt."""
        ...

    async def create_prompt(
        self,
        prompt: str,
        variables: list[str] | None = None,
    ) -> Prompt:
        """Create prompt."""
        ...

    async def update_prompt(
        self,
        prompt_id: str,
        prompt: str,
        version: int,
        variables: list[str] | None = None,
        set_as_default: bool = True,
    ) -> Prompt:
        """Update prompt."""
        ...

    async def delete_prompt(
        self,
        prompt_id: str,
    ) -> None:
        """Delete prompt."""
        ...

    async def set_default_version(
        self,
        prompt_id: str,
        version: int,
    ) -> Prompt:
        """Set prompt version."""
        ...

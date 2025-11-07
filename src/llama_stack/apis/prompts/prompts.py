# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import re
import secrets
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field, field_validator, model_validator

from llama_stack.apis.common.tracing import telemetry_traceable
from llama_stack.apis.version import LLAMA_STACK_API_V1
from llama_stack.schema_utils import json_schema_type, webmethod


@json_schema_type
class Prompt(BaseModel):
    """A prompt resource representing a stored OpenAI Compatible prompt template in Llama Stack.

    :param prompt: The system prompt text with variable placeholders. Variables are only supported when using the Responses API.
    :param version: Version (integer starting at 1, incremented on save)
    :param prompt_id: Unique identifier formatted as 'pmpt_<48-digit-hash>'
    :param variables: List of prompt variable names that can be used in the prompt template
    :param is_default: Boolean indicating whether this version is the default version for this prompt
    """

    prompt: str | None = Field(default=None, description="The system prompt with variable placeholders")
    version: int = Field(description="Version (integer starting at 1, incremented on save)", ge=1)
    prompt_id: str = Field(description="Unique identifier in format 'pmpt_<48-digit-hash>'")
    variables: list[str] = Field(
        default_factory=list, description="List of variable names that can be used in the prompt template"
    )
    is_default: bool = Field(
        default=False, description="Boolean indicating whether this version is the default version"
    )

    @field_validator("prompt_id")
    @classmethod
    def validate_prompt_id(cls, prompt_id: str) -> str:
        if not isinstance(prompt_id, str):
            raise TypeError("prompt_id must be a string in format 'pmpt_<48-digit-hash>'")

        if not prompt_id.startswith("pmpt_"):
            raise ValueError("prompt_id must start with 'pmpt_' prefix")

        hex_part = prompt_id[5:]
        if len(hex_part) != 48:
            raise ValueError("prompt_id must be in format 'pmpt_<48-digit-hash>' (48 lowercase hex chars)")

        for char in hex_part:
            if char not in "0123456789abcdef":
                raise ValueError("prompt_id hex part must contain only lowercase hex characters [0-9a-f]")

        return prompt_id

    @field_validator("version")
    @classmethod
    def validate_version(cls, prompt_version: int) -> int:
        if prompt_version < 1:
            raise ValueError("version must be >= 1")
        return prompt_version

    @model_validator(mode="after")
    def validate_prompt_variables(self):
        """Validate that all variables used in the prompt are declared in the variables list."""
        if not self.prompt:
            return self

        prompt_variables = set(re.findall(r"{{\s*(\w+)\s*}}", self.prompt))
        declared_variables = set(self.variables)

        undeclared = prompt_variables - declared_variables
        if undeclared:
            raise ValueError(f"Prompt contains undeclared variables: {sorted(undeclared)}")

        return self

    @classmethod
    def generate_prompt_id(cls) -> str:
        # Generate 48 hex characters (24 bytes)
        random_bytes = secrets.token_bytes(24)
        hex_string = random_bytes.hex()
        return f"pmpt_{hex_string}"


class ListPromptsResponse(BaseModel):
    """Response model to list prompts."""

    data: list[Prompt]


@runtime_checkable
@telemetry_traceable
class Prompts(Protocol):
    """Prompts

    Protocol for prompt management operations."""

    @webmethod(route="/prompts", method="GET", level=LLAMA_STACK_API_V1)
    async def list_prompts(self) -> ListPromptsResponse:
        """List all prompts.

        :returns: A ListPromptsResponse containing all prompts.
        """
        ...

    @webmethod(route="/prompts/{prompt_id}/versions", method="GET", level=LLAMA_STACK_API_V1)
    async def list_prompt_versions(
        self,
        prompt_id: str,
    ) -> ListPromptsResponse:
        """List prompt versions.

        List all versions of a specific prompt.

        :param prompt_id: The identifier of the prompt to list versions for.
        :returns: A ListPromptsResponse containing all versions of the prompt.
        """
        ...

    @webmethod(route="/prompts/{prompt_id}", method="GET", level=LLAMA_STACK_API_V1)
    async def get_prompt(
        self,
        prompt_id: str,
        version: int | None = None,
    ) -> Prompt:
        """Get prompt.

        Get a prompt by its identifier and optional version.

        :param prompt_id: The identifier of the prompt to get.
        :param version: The version of the prompt to get (defaults to latest).
        :returns: A Prompt resource.
        """
        ...

    @webmethod(route="/prompts", method="POST", level=LLAMA_STACK_API_V1)
    async def create_prompt(
        self,
        prompt: str,
        variables: list[str] | None = None,
    ) -> Prompt:
        """Create prompt.

        Create a new prompt.

        :param prompt: The prompt text content with variable placeholders.
        :param variables: List of variable names that can be used in the prompt template.
        :returns: The created Prompt resource.
        """
        ...

    @webmethod(route="/prompts/{prompt_id}", method="PUT", level=LLAMA_STACK_API_V1)
    async def update_prompt(
        self,
        prompt_id: str,
        prompt: str,
        version: int,
        variables: list[str] | None = None,
        set_as_default: bool = True,
    ) -> Prompt:
        """Update prompt.

        Update an existing prompt (increments version).

        :param prompt_id: The identifier of the prompt to update.
        :param prompt: The updated prompt text content.
        :param version: The current version of the prompt being updated.
        :param variables: Updated list of variable names that can be used in the prompt template.
        :param set_as_default: Set the new version as the default (default=True).
        :returns: The updated Prompt resource with incremented version.
        """
        ...

    @webmethod(route="/prompts/{prompt_id}", method="DELETE", level=LLAMA_STACK_API_V1)
    async def delete_prompt(
        self,
        prompt_id: str,
    ) -> None:
        """Delete prompt.

        Delete a prompt.

        :param prompt_id: The identifier of the prompt to delete.
        """
        ...

    @webmethod(route="/prompts/{prompt_id}/set-default-version", method="PUT", level=LLAMA_STACK_API_V1)
    async def set_default_version(
        self,
        prompt_id: str,
        version: int,
    ) -> Prompt:
        """Set prompt version.

        Set which version of a prompt should be the default in get_prompt (latest).

        :param prompt_id: The identifier of the prompt.
        :param version: The version to set as default.
        :returns: The prompt with the specified version now set as default.
        """
        ...

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import re
import secrets

from pydantic import BaseModel, Field, field_validator, model_validator

from llama_stack.schema_utils import json_schema_type


@json_schema_type
class Prompt(BaseModel):
    """A prompt resource representing a stored OpenAI Compatible prompt template in Llama Stack."""

    prompt: str | None = Field(
        default=None,
        description="The system prompt text with variable placeholders. Variables are only supported when using the Responses API.",
    )
    version: int = Field(description="Version (integer starting at 1, incremented on save).", ge=1)
    prompt_id: str = Field(description="Unique identifier formatted as 'pmpt_<48-digit-hash>'.")
    variables: list[str] = Field(
        default_factory=list, description="List of prompt variable names that can be used in the prompt template."
    )
    is_default: bool = Field(
        default=False, description="Boolean indicating whether this version is the default version for this prompt."
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

    data: list[Prompt] = Field(description="List of prompt resources.")


@json_schema_type
class CreatePromptRequest(BaseModel):
    """Request model for creating a new prompt."""

    prompt: str = Field(..., description="The prompt text content with variable placeholders.")
    variables: list[str] | None = Field(
        default=None, description="List of variable names that can be used in the prompt template."
    )


@json_schema_type
class UpdatePromptRequest(BaseModel):
    """Request model for updating an existing prompt."""

    prompt: str = Field(..., description="The updated prompt text content.")
    version: int = Field(..., description="The current version of the prompt being updated.")
    variables: list[str] | None = Field(
        default=None, description="Updated list of variable names that can be used in the prompt template."
    )
    set_as_default: bool = Field(default=True, description="Set the new version as the default (default=True).")


@json_schema_type
class SetDefaultVersionRequest(BaseModel):
    """Request model for setting a prompt version as default."""

    version: int = Field(..., description="The version to set as default.")

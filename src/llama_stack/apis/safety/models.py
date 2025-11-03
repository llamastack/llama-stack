# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from llama_stack.apis.inference import OpenAIMessageParam
from llama_stack.schema_utils import json_schema_type


@json_schema_type
class ModerationObjectResults(BaseModel):
    """A moderation object."""

    flagged: bool = Field(..., description="Whether any of the below categories are flagged.")
    categories: dict[str, bool] | None = Field(
        default=None, description="A list of the categories, and whether they are flagged or not."
    )
    category_applied_input_types: dict[str, list[str]] | None = Field(
        default=None,
        description="A list of the categories along with the input type(s) that the score applies to.",
    )
    category_scores: dict[str, float] | None = Field(
        default=None, description="A list of the categories along with their scores as predicted by model."
    )
    user_message: str | None = Field(default=None, description="User message.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata.")


@json_schema_type
class ModerationObject(BaseModel):
    """A moderation object."""

    id: str = Field(..., description="The unique identifier for the moderation request.")
    model: str = Field(..., description="The model used to generate the moderation results.")
    results: list[ModerationObjectResults] = Field(..., description="A list of moderation objects.")


@json_schema_type
class ViolationLevel(Enum):
    """Severity level of a safety violation.

    :cvar INFO: Informational level violation that does not require action
    :cvar WARN: Warning level violation that suggests caution but allows continuation
    :cvar ERROR: Error level violation that requires blocking or intervention
    """

    INFO = "info"
    WARN = "warn"
    ERROR = "error"


@json_schema_type
class SafetyViolation(BaseModel):
    """Details of a safety violation detected by content moderation."""

    violation_level: ViolationLevel = Field(..., description="Severity level of the violation.")
    user_message: str | None = Field(default=None, description="Message to convey to the user about the violation.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata including specific violation codes for debugging and telemetry.",
    )


@json_schema_type
class RunShieldResponse(BaseModel):
    """Response from running a safety shield."""

    violation: SafetyViolation | None = Field(
        default=None, description="Safety violation detected by the shield, if any."
    )


@json_schema_type
class RunShieldRequest(BaseModel):
    """Request model for running a shield."""

    shield_id: str = Field(..., description="The identifier of the shield to run.")
    messages: list[OpenAIMessageParam] = Field(..., description="The messages to run the shield on.")
    params: dict[str, Any] = Field(..., description="The parameters of the shield.")


@json_schema_type
class RunModerationRequest(BaseModel):
    """Request model for running moderation."""

    input: str | list[str] = Field(
        ...,
        description="Input (or inputs) to classify. Can be a single string, an array of strings, or an array of multi-modal input objects similar to other models.",
    )
    model: str | None = Field(default=None, description="The content moderation model you would like to use.")

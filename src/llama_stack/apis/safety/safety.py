# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from llama_stack.apis.common.tracing import telemetry_traceable
from llama_stack.apis.inference import OpenAIMessageParam
from llama_stack.apis.shields import Shield
from llama_stack.apis.version import LLAMA_STACK_API_V1
from llama_stack.schema_utils import json_schema_type, webmethod


@json_schema_type
class ModerationObjectResults(BaseModel):
    """A moderation object.
    :param flagged: Whether any of the below categories are flagged.
    :param categories: A list of the categories, and whether they are flagged or not.
    :param category_applied_input_types: A list of the categories along with the input type(s) that the score applies to.
    :param category_scores: A list of the categories along with their scores as predicted by model.
    """

    flagged: bool
    categories: dict[str, bool] | None = None
    category_applied_input_types: dict[str, list[str]] | None = None
    category_scores: dict[str, float] | None = None
    user_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@json_schema_type
class ModerationObject(BaseModel):
    """A moderation object.
    :param id: The unique identifier for the moderation request.
    :param model: The model used to generate the moderation results.
    :param results: A list of moderation objects
    """

    id: str
    model: str
    results: list[ModerationObjectResults]


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
    """Details of a safety violation detected by content moderation.

    :param violation_level: Severity level of the violation
    :param user_message: (Optional) Message to convey to the user about the violation
    :param metadata: Additional metadata including specific violation codes for debugging and telemetry
    """

    violation_level: ViolationLevel

    # what message should you convey to the user
    user_message: str | None = None

    # additional metadata (including specific violation codes) more for
    # debugging, telemetry
    metadata: dict[str, Any] = Field(default_factory=dict)


@json_schema_type
class RunShieldResponse(BaseModel):
    """Response from running a safety shield.

    :param violation: (Optional) Safety violation detected by the shield, if any
    """

    violation: SafetyViolation | None = None


class ShieldStore(Protocol):
    async def get_shield(self, identifier: str) -> Shield: ...


@runtime_checkable
@telemetry_traceable
class Safety(Protocol):
    """Safety

    OpenAI-compatible Moderations API.
    """

    shield_store: ShieldStore

    @webmethod(route="/safety/run-shield", method="POST", level=LLAMA_STACK_API_V1)
    async def run_shield(
        self,
        shield_id: str,
        messages: list[OpenAIMessageParam],
        params: dict[str, Any],
    ) -> RunShieldResponse:
        """Run shield.

        Run a shield.

        :param shield_id: The identifier of the shield to run.
        :param messages: The messages to run the shield on.
        :param params: The parameters of the shield.
        :returns: A RunShieldResponse.
        """
        ...

    @webmethod(route="/moderations", method="POST", level=LLAMA_STACK_API_V1)
    async def run_moderation(self, input: str | list[str], model: str | None = None) -> ModerationObject:
        """Create moderation.

        Classifies if text and/or image inputs are potentially harmful.
        :param input: Input (or inputs) to classify.
        Can be a single string, an array of strings, or an array of multi-modal input objects similar to other models.
        :param model: (Optional) The content moderation model you would like to use.
        :returns: A moderation object.
        """
        ...

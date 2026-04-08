# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pydantic models for the Google Interactions API.

These models define the request and response shapes for the /v1/interactions endpoint,
following the Google Interactions API specification.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# -- Content items --


class GoogleTextContent(BaseModel):
    """A text content item."""

    type: Literal["text"] = "text"
    text: str


# -- Conversation turns --


class GoogleInputTurn(BaseModel):
    """A conversation turn in the input."""

    role: Literal["user", "model"]
    content: list[GoogleTextContent] = Field(
        ...,
        description="Content items for this turn.",
    )


GoogleInputItem = Annotated[
    GoogleInputTurn,
    Field(description="A conversation turn."),
]


# -- Generation config --


class GoogleGenerationConfig(BaseModel):
    """Generation parameters for the Interactions API."""

    temperature: float | None = Field(default=None, ge=0.0, le=2.0, description="Sampling temperature.")
    top_k: int | None = Field(default=None, ge=1, description="Top-k sampling parameter.")
    top_p: float | None = Field(default=None, ge=0.0, le=1.0, description="Nucleus sampling parameter.")
    max_output_tokens: int | None = Field(default=None, ge=1, description="Maximum number of tokens to generate.")


# -- Request models --


class GoogleCreateInteractionRequest(BaseModel):
    """Request body for POST /v1/interactions."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(..., description="The model to use for generation.")
    input: str | list[GoogleInputItem] = Field(
        ...,
        description="Prompt string or list of conversation turns.",
    )
    system_instruction: str | None = Field(
        default=None,
        description="System prompt.",
    )
    generation_config: GoogleGenerationConfig | None = Field(
        default=None,
        description="Generation parameters.",
    )
    stream: bool | None = Field(default=False, description="Whether to stream the response via SSE.")
    response_modalities: list[str] | None = Field(
        default=None,
        description="Accepted response modalities (e.g. ['TEXT']). Accepted for compatibility, ignored in v1.",
    )


# -- Response models --


class GoogleTextOutput(BaseModel):
    """A text output item."""

    type: Literal["text"] = "text"
    text: str


class GoogleUsage(BaseModel):
    """Token usage statistics."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class GoogleInteractionResponse(BaseModel):
    """Response from POST /v1/interactions (non-streaming)."""

    id: str = Field(..., description="Unique interaction ID.")
    status: Literal["completed"] = "completed"
    outputs: list[GoogleTextOutput] = Field(..., description="Response output items.")
    usage: GoogleUsage = Field(default_factory=GoogleUsage)


# -- Streaming event models --


class InteractionStartEvent(BaseModel):
    """First event in a streaming response."""

    event_type: Literal["interaction.start"] = "interaction.start"
    id: str
    status: str = "in_progress"


class ContentStartEvent(BaseModel):
    """Signals the start of a new content block."""

    event_type: Literal["content.start"] = "content.start"
    index: int
    type: Literal["text"] = "text"


class _TextDelta(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ContentDeltaEvent(BaseModel):
    """A delta within a content block."""

    event_type: Literal["content.delta"] = "content.delta"
    index: int
    delta: _TextDelta


class ContentStopEvent(BaseModel):
    """Signals the end of a content block."""

    event_type: Literal["content.stop"] = "content.stop"
    index: int


class InteractionCompleteEvent(BaseModel):
    """Final event in a streaming response."""

    event_type: Literal["interaction.complete"] = "interaction.complete"
    id: str
    status: Literal["completed"] = "completed"
    usage: GoogleUsage = Field(default_factory=GoogleUsage)


GoogleStreamEvent = (
    InteractionStartEvent | ContentStartEvent | ContentDeltaEvent | ContentStopEvent | InteractionCompleteEvent
)


# -- Error response --


class _GoogleErrorDetail(BaseModel):
    code: int
    message: str


class GoogleErrorResponse(BaseModel):
    """Google-format error response."""

    error: _GoogleErrorDetail

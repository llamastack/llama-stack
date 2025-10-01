# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
from enum import Enum, StrEnum
from io import BytesIO
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

# The goal is that these set of types are relevant for all Llama models.
# That isn't the current state yet -- e.g., BuiltinTool is somewhat specific to
# the llama3 series of models.


class Role(Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class BuiltinTool(Enum):
    brave_search = "brave_search"
    wolfram_alpha = "wolfram_alpha"
    photogen = "photogen"
    code_interpreter = "code_interpreter"


Primitive = str | int | float | bool | None
RecursiveType = Primitive | list[Primitive] | dict[str, Primitive]


class ToolCall(BaseModel):
    call_id: str
    tool_name: BuiltinTool | str
    arguments: str

    @field_validator("tool_name", mode="before")
    @classmethod
    def validate_field(cls, v):
        if isinstance(v, str):
            try:
                return BuiltinTool(v)
            except ValueError:
                return v
        return v


class ToolPromptFormat(Enum):
    """Prompt format for calling custom / zero shot tools.

    :cvar json: JSON format for calling tools. It takes the form:
        {
            "type": "function",
            "function" : {
                "name": "function_name",
                "description": "function_description",
                "parameters": {...}
            }
        }
    :cvar function_tag: Function tag format, pseudo-XML. This looks like:
        <function=function_name>(parameters)</function>

    :cvar python_list: Python list. The output is a valid Python expression that can be
        evaluated to a list. Each element in the list is a function call. Example:
        ["function_name(param1, param2)", "function_name(param1, param2)"]
    """

    json = "json"
    function_tag = "function_tag"
    python_list = "python_list"


class StopReason(Enum):
    end_of_turn = "end_of_turn"
    end_of_message = "end_of_message"
    out_of_tokens = "out_of_tokens"


class ToolDefinition(BaseModel):
    tool_name: BuiltinTool | str
    description: str | None = None
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None

    @field_validator("tool_name", mode="before")
    @classmethod
    def validate_field(cls, v):
        if isinstance(v, str):
            try:
                return BuiltinTool(v)
            except ValueError:
                return v
        return v


class RawMediaItem(BaseModel):
    type: Literal["image"] = "image"
    data: bytes | BytesIO

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer("data")
    def serialize_data(self, data: bytes | None, _info):
        if data is None:
            return None
        return base64.b64encode(data).decode("utf-8")

    @field_validator("data", mode="before")
    @classmethod
    def validate_data(cls, v):
        if isinstance(v, str):
            return base64.b64decode(v)
        return v


class RawTextItem(BaseModel):
    type: Literal["text"] = "text"
    text: str


RawContentItem = Annotated[RawTextItem | RawMediaItem, Field(discriminator="type")]

RawContent = str | RawContentItem | list[RawContentItem]


class RawMessage(BaseModel):
    role: Literal["user"] | Literal["system"] | Literal["tool"] | Literal["assistant"]
    content: RawContent

    # This is for RAG but likely should be absorbed into content
    context: RawContent | None = None

    # These are for the output message coming from the assistant
    stop_reason: StopReason | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class GenerationResult(BaseModel):
    token: int
    text: str
    logprobs: list[float] | None = None

    source: Literal["input"] | Literal["output"]

    # index within the batch
    batch_idx: int
    # whether generation for this item is already finished. note that tokens can
    # get returned even afterwards since other items in the batch can still be generating tokens
    finished: bool
    # because a batch is parallel processed, useful decoding for one item can correspond to processing
    # pad tokens or tokens beyond EOS for other items. we could have decided to return None for this case
    # but it's more convenient to return a list of GenerationResult and filter out the ignored tokens
    ignore_token: bool


class QuantizationMode(StrEnum):
    none = "none"
    fp8_mixed = "fp8_mixed"
    int4_mixed = "int4_mixed"

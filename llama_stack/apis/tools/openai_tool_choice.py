# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Literal

from pydantic import BaseModel

from llama_stack.schema_utils import json_schema_type, register_schema

ToolChoiceOptions = Literal["none", "auto", "required"]
register_schema(ToolChoiceOptions, name="ToolChoiceOptions")


@json_schema_type
class ToolChoiceTypes(BaseModel):
    type: Literal[
        "file_search",
        "web_search_preview",
        "computer_use_preview",
        "web_search_preview_2025_03_11",
        "image_generation",
        "code_interpreter",
    ]
    """The type of hosted tool the model should use.

    Allowed values are:

    - `file_search`
    - `web_search_preview`
    - `computer_use_preview`
    - `code_interpreter`
    - `image_generation`
    """


@json_schema_type
class ToolChoiceAllowed(BaseModel):
    mode: Literal["auto", "required"]
    """Constrains the tools available to the model to a pre-defined set.

    `auto` allows the model to pick from among the allowed tools and generate a
    message.

    `required` requires the model to call one or more of the allowed tools.
    """

    tools: list[dict[str, object]]
    """A list of tool definitions that the model should be allowed to call.

    For the Responses API, the list of tool definitions might look like:

    ```json
    [
      { "type": "function", "name": "get_weather" },
      { "type": "mcp", "server_label": "deepwiki" },
      { "type": "image_generation" }
    ]
    ```
    """

    type: Literal["allowed_tools"] = "allowed_tools"
    """Allowed tool configuration type. Always `allowed_tools`."""


@json_schema_type
class ToolChoiceFunction(BaseModel):
    name: str
    """The name of the function to call."""

    type: Literal["function"] = "function"
    """For function calling, the type is always `function`."""


@json_schema_type
class ToolChoiceMcp(BaseModel):
    server_label: str
    """The label of the MCP server to use."""

    type: Literal["mcp"] = "mcp"
    """For MCP tools, the type is always `mcp`."""

    name: str | None = None
    """The name of the tool to call on the server."""


@json_schema_type
class ToolChoiceCustom(BaseModel):
    name: str
    """The name of the custom tool to call."""

    type: Literal["custom"] = "custom"
    """For custom tool calling, the type is always `custom`."""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from pydantic import TypeAdapter, ValidationError

from llama_stack.apis.agents.openai_responses import OpenAIResponsesToolChoice
from llama_stack.apis.tools.openai_tool_choice import (
    ToolChoiceAllowed,
    ToolChoiceCustom,
    ToolChoiceFunction,
    ToolChoiceMcp,
    ToolChoiceOptions,
    ToolChoiceTypes,
)


def test_tool_choice_discriminated_options():
    adapter = TypeAdapter(OpenAIResponsesToolChoice)

    cases = [
        ({"type": "function", "name": "search"}, ToolChoiceFunction, "function"),
        ({"type": "mcp", "server_label": "deepwiki"}, ToolChoiceMcp, "mcp"),
        ({"type": "custom", "name": "my_tool"}, ToolChoiceCustom, "custom"),
        (
            {
                "type": "allowed_tools",
                "mode": "auto",
                "tools": [{"type": "function", "name": "foo"}],
            },
            ToolChoiceAllowed,
            "allowed_tools",
        ),
    ]

    for payload, expected_cls, expected_type in cases:
        obj = adapter.validate_python(payload)
        assert isinstance(obj, expected_cls)
        assert obj.type == expected_type

        dumped = obj.model_dump()
        reparsed = adapter.validate_python(dumped)
        assert isinstance(reparsed, expected_cls)
        assert reparsed.model_dump() == dumped


def test_tool_choice_literal_options():
    adapter = TypeAdapter(OpenAIResponsesToolChoice)
    options_adapter = TypeAdapter(ToolChoiceOptions)

    for v in ("none", "auto", "required"):
        # Validate via the specific literal adapter
        assert options_adapter.validate_python(v) == v
        # And via the top-level union adapter
        assert adapter.validate_python(v) == v


def test_tool_choice_rejects_invalid_value():
    adapter = TypeAdapter(OpenAIResponsesToolChoice)

    with pytest.raises(ValidationError):
        adapter.validate_python("invalid")
    with pytest.raises(ValidationError):
        adapter.validate_python({"type": "unknown_variant"})


def test_tool_choice_types_accepts_each_variant_value():
    adapter = TypeAdapter(OpenAIResponsesToolChoice)

    allowed_values = [
        "file_search",
        "web_search_preview",
        "computer_use_preview",
        "web_search_preview_2025_03_11",
        "image_generation",
        "code_interpreter",
    ]

    for v in allowed_values:
        obj = adapter.validate_python({"type": v})
        assert isinstance(obj, ToolChoiceTypes)
        assert obj.type == v
        assert obj.model_dump() == {"type": v}


def test_tool_choice_rejects_invalid_discriminator_value():
    adapter = TypeAdapter(OpenAIResponsesToolChoice)
    with pytest.raises(ValidationError):
        adapter.validate_python({"type": "unknown_variant"})


def test_tool_choice_rejects_missing_required_fields():
    adapter = TypeAdapter(OpenAIResponsesToolChoice)
    # Missing "name" for function
    with pytest.raises(ValidationError):
        adapter.validate_python({"type": "function"})

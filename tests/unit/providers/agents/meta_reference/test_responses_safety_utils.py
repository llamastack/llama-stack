# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock

import pytest

from llama_stack.apis.agents.agents import ResponseGuardrailSpec
from llama_stack.providers.inline.agents.meta_reference.responses.openai_responses import (
    OpenAIResponsesImpl,
)
from llama_stack.providers.inline.agents.meta_reference.responses.utils import (
    extract_guardrail_ids,
    extract_text_content,
)


@pytest.fixture
def mock_apis():
    """Create mock APIs for testing."""
    return {
        "inference_api": AsyncMock(),
        "tool_groups_api": AsyncMock(),
        "tool_runtime_api": AsyncMock(),
        "responses_store": AsyncMock(),
        "vector_io_api": AsyncMock(),
        "conversations_api": AsyncMock(),
        "safety_api": AsyncMock(),
    }


@pytest.fixture
def responses_impl(mock_apis):
    """Create OpenAIResponsesImpl instance with mocked dependencies."""
    return OpenAIResponsesImpl(**mock_apis)


def test_extract_guardrail_ids_from_strings(responses_impl):
    """Test extraction from simple string guardrail IDs."""
    guardrails = ["llama-guard", "content-filter", "nsfw-detector"]
    result = extract_guardrail_ids(guardrails)
    assert result == ["llama-guard", "content-filter", "nsfw-detector"]


def test_extract_guardrail_ids_from_objects(responses_impl):
    """Test extraction from ResponseGuardrailSpec objects."""
    guardrails = [
        ResponseGuardrailSpec(type="llama-guard"),
        ResponseGuardrailSpec(type="content-filter"),
    ]
    result = extract_guardrail_ids(guardrails)
    assert result == ["llama-guard", "content-filter"]


def test_extract_guardrail_ids_mixed_formats(responses_impl):
    """Test extraction from mixed string and object formats."""
    guardrails = [
        "llama-guard",
        ResponseGuardrailSpec(type="content-filter"),
        "nsfw-detector",
    ]
    result = extract_guardrail_ids(guardrails)
    assert result == ["llama-guard", "content-filter", "nsfw-detector"]


def test_extract_guardrail_ids_none_input(responses_impl):
    """Test extraction with None input."""
    result = extract_guardrail_ids(None)
    assert result == []


def test_extract_guardrail_ids_empty_list(responses_impl):
    """Test extraction with empty list."""
    result = extract_guardrail_ids([])
    assert result == []


def test_extract_guardrail_ids_unknown_format(responses_impl):
    """Test extraction with unknown guardrail format raises ValueError."""
    # Create an object that's neither string nor ResponseGuardrailSpec
    unknown_object = {"invalid": "format"}  # Plain dict, not ResponseGuardrailSpec
    guardrails = ["valid-guardrail", unknown_object, "another-guardrail"]
    with pytest.raises(ValueError, match="Unknown guardrail format.*expected str or ResponseGuardrailSpec"):
        extract_guardrail_ids(guardrails)


def test_extract_text_content_string(responses_impl):
    """Test extraction from simple string content."""
    content = "Hello world"
    result = extract_text_content(content)
    assert result == "Hello world"


def test_extract_text_content_list_with_text(responses_impl):
    """Test extraction from list content with text parts."""
    content = [
        MagicMock(text="Hello "),
        MagicMock(text="world"),
    ]
    result = extract_text_content(content)
    assert result == "Hello  world"


def test_extract_text_content_list_with_refusal(responses_impl):
    """Test extraction skips refusal parts."""
    # Create text parts
    text_part1 = MagicMock()
    text_part1.text = "Hello"

    text_part2 = MagicMock()
    text_part2.text = "world"

    # Create refusal part (no text attribute)
    refusal_part = MagicMock()
    refusal_part.type = "refusal"
    refusal_part.refusal = "Blocked"
    del refusal_part.text  # Remove text attribute

    content = [text_part1, refusal_part, text_part2]
    result = extract_text_content(content)
    assert result == "Hello world"


def test_extract_text_content_empty_list(responses_impl):
    """Test extraction from empty list returns None."""
    content = []
    result = extract_text_content(content)
    assert result is None


def test_extract_text_content_no_text_parts(responses_impl):
    """Test extraction with no text parts returns None."""
    # Create image part (no text attribute)
    image_part = MagicMock()
    image_part.type = "image"
    image_part.image_url = "http://example.com"

    # Create refusal part (no text attribute)
    refusal_part = MagicMock()
    refusal_part.type = "refusal"
    refusal_part.refusal = "Blocked"

    # Explicitly remove text attributes to simulate non-text parts
    if hasattr(image_part, "text"):
        delattr(image_part, "text")
    if hasattr(refusal_part, "text"):
        delattr(refusal_part, "text")

    content = [image_part, refusal_part]
    result = extract_text_content(content)
    assert result is None


def test_extract_text_content_none_input(responses_impl):
    """Test extraction with None input returns None."""
    result = extract_text_content(None)
    assert result is None

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock

import pytest

from llama_stack.apis.agents.agents import ResponseShieldSpec
from llama_stack.providers.inline.agents.meta_reference.responses.openai_responses import (
    OpenAIResponsesImpl,
)
from llama_stack.providers.inline.agents.meta_reference.responses.utils import (
    extract_shield_ids,
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


def test_extract_shield_ids_from_strings(responses_impl):
    """Test extraction from simple string shield IDs."""
    shields = ["llama-guard", "content-filter", "nsfw-detector"]
    result = extract_shield_ids(shields)
    assert result == ["llama-guard", "content-filter", "nsfw-detector"]


def test_extract_shield_ids_from_objects(responses_impl):
    """Test extraction from ResponseShieldSpec objects."""
    shields = [
        ResponseShieldSpec(type="llama-guard"),
        ResponseShieldSpec(type="content-filter"),
    ]
    result = extract_shield_ids(shields)
    assert result == ["llama-guard", "content-filter"]


def test_extract_shield_ids_mixed_formats(responses_impl):
    """Test extraction from mixed string and object formats."""
    shields = [
        "llama-guard",
        ResponseShieldSpec(type="content-filter"),
        "nsfw-detector",
    ]
    result = extract_shield_ids(shields)
    assert result == ["llama-guard", "content-filter", "nsfw-detector"]


def test_extract_shield_ids_none_input(responses_impl):
    """Test extraction with None input."""
    result = extract_shield_ids(None)
    assert result == []


def test_extract_shield_ids_empty_list(responses_impl):
    """Test extraction with empty list."""
    result = extract_shield_ids([])
    assert result == []


def test_extract_shield_ids_unknown_format(responses_impl):
    """Test extraction with unknown shield format raises ValueError."""
    # Create an object that's neither string nor ResponseShieldSpec
    unknown_object = {"invalid": "format"}  # Plain dict, not ResponseShieldSpec
    shields = ["valid-shield", unknown_object, "another-shield"]
    with pytest.raises(ValueError, match="Unknown shield format.*expected str or ResponseShieldSpec"):
        extract_shield_ids(shields)


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

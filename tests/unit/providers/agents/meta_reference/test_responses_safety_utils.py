# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock

import pytest

from llama_stack.apis.agents.agents import ResponseGuardrailSpec
from llama_stack.providers.inline.agents.meta_reference.responses.openai_responses import (
    OpenAIResponsesImpl,
)
from llama_stack.providers.inline.agents.meta_reference.responses.utils import (
    extract_guardrail_ids,
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

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from unittest.mock import AsyncMock, patch

import pytest

from llama_stack.providers.remote.safety.nvidia.config import GuardrailsApiMode, NVIDIASafetyConfig
from llama_stack.providers.remote.safety.nvidia.nvidia import NVIDIASafetyAdapter
from llama_stack_api import (
    OpenAIAssistantMessageParam,
    OpenAIUserMessageParam,
    ResourceType,
    RunShieldResponse,
    Shield,
    ViolationLevel,
)


class FakeNVIDIASafetyAdapter(NVIDIASafetyAdapter):
    """Test implementation that provides the required shield_store."""

    def __init__(self, config: NVIDIASafetyConfig, shield_store):
        super().__init__(config)
        self.shield_store = shield_store


@pytest.fixture
def nvidia_adapter():
    """Set up the NVIDIASafetyAdapter for testing with OpenAI API mode."""
    os.environ["NVIDIA_GUARDRAILS_URL"] = "http://nemo.test"

    config = NVIDIASafetyConfig(
        guardrails_service_url=os.environ["NVIDIA_GUARDRAILS_URL"],
        api_mode=GuardrailsApiMode.OPENAI,
    )

    shield_store = AsyncMock()
    shield_store.get_shield = AsyncMock()

    return FakeNVIDIASafetyAdapter(config=config, shield_store=shield_store)


@pytest.fixture
def nvidia_adapter_microservice():
    """Set up the NVIDIASafetyAdapter for testing with Microservice API mode."""
    os.environ["NVIDIA_GUARDRAILS_URL"] = "http://nemo.test"

    config = NVIDIASafetyConfig(
        guardrails_service_url=os.environ["NVIDIA_GUARDRAILS_URL"],
        api_mode=GuardrailsApiMode.MICROSERVICE,
    )

    shield_store = AsyncMock()
    shield_store.get_shield = AsyncMock()

    return FakeNVIDIASafetyAdapter(config=config, shield_store=shield_store)


@pytest.fixture
def mock_guardrails_post():
    """Mock the HTTP request methods."""
    with patch("llama_stack.providers.remote.safety.nvidia.nvidia.NeMoGuardrails._guardrails_post") as mock_post:
        mock_post.return_value = {"status": "allowed"}
        yield mock_post


async def test_register_shield_with_valid_id(nvidia_adapter):
    adapter = nvidia_adapter

    shield = Shield(
        provider_id="nvidia",
        type=ResourceType.shield,
        identifier="test-shield",
        provider_resource_id="test-model",
    )

    # Register the shield
    await adapter.register_shield(shield)


async def test_register_shield_without_id(nvidia_adapter):
    adapter = nvidia_adapter

    shield = Shield(
        provider_id="nvidia",
        type=ResourceType.shield,
        identifier="test-shield",
        provider_resource_id="",
    )

    # Register the shield should raise a ValueError
    with pytest.raises(ValueError):
        await adapter.register_shield(shield)


async def test_run_shield_allowed(nvidia_adapter, mock_guardrails_post):
    adapter = nvidia_adapter

    # Set up the shield
    shield_id = "test-shield"
    shield = Shield(
        provider_id="nvidia",
        type=ResourceType.shield,
        identifier=shield_id,
        provider_resource_id="test-model",
    )
    adapter.shield_store.get_shield.return_value = shield

    # Mock Guardrails API response
    mock_guardrails_post.return_value = {"status": "allowed"}

    # Run the shield
    messages = [
        OpenAIUserMessageParam(content="Hello, how are you?"),
        OpenAIAssistantMessageParam(
            content="I'm doing well, thank you for asking!",
            tool_calls=[],
        ),
    ]
    result = await adapter.run_shield(shield_id, messages)

    # Verify the shield store was called
    adapter.shield_store.get_shield.assert_called_once_with(shield_id)

    # Verify the Guardrails API was called correctly
    mock_guardrails_post.assert_called_once_with(
        path="/v1/guardrail/chat/completions",
        data={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
            ],
            "guardrails": {
                "config_id": "self-check",
            },
            "temperature": 1.0,
        },
    )

    # Verify the result
    assert isinstance(result, RunShieldResponse)
    assert result.violation is None


async def test_run_shield_blocked_with_error_object(nvidia_adapter, mock_guardrails_post):
    """Test that shield correctly detects blocks via NeMo Guardrails error object format."""
    adapter = nvidia_adapter

    shield_id = "test-shield"
    shield = Shield(
        provider_id="nvidia",
        type=ResourceType.shield,
        identifier=shield_id,
        provider_resource_id="test-model",
    )
    adapter.shield_store.get_shield.return_value = shield

    # Mock Guardrails API response with error object (NeMo Guardrails v25.06 format)
    mock_guardrails_post.return_value = {
        "error": {
            "message": "Blocked by content-moderation rails.",
            "type": "guardrails_violation",
            "param": "content-moderation",
            "code": "content_blocked",
        }
    }

    messages = [
        OpenAIUserMessageParam(content="Hello, how are you?"),
        OpenAIAssistantMessageParam(
            content="I'm doing well, thank you for asking!",
            tool_calls=[],
        ),
    ]
    result = await adapter.run_shield(shield_id, messages)

    adapter.shield_store.get_shield.assert_called_once_with(shield_id)

    mock_guardrails_post.assert_called_once_with(
        path="/v1/guardrail/chat/completions",
        data={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
            ],
            "guardrails": {
                "config_id": "self-check",
            },
            "temperature": 1.0,
        },
    )

    assert result.violation is not None
    assert isinstance(result, RunShieldResponse)
    assert result.violation.user_message == "Blocked by content-moderation rails."
    assert result.violation.violation_level == ViolationLevel.ERROR
    assert result.violation.metadata == {
        "error_type": "guardrails_violation",
        "error_code": "content_blocked",
        "rail_name": "content-moderation",
    }


async def test_run_shield_blocked_with_status(nvidia_adapter, mock_guardrails_post):
    """Test that shield correctly detects blocks via status field (legacy format)."""
    adapter = nvidia_adapter

    # Set up the shield
    shield_id = "test-shield"
    shield = Shield(
        provider_id="nvidia",
        type=ResourceType.shield,
        identifier=shield_id,
        provider_resource_id="test-model",
    )
    adapter.shield_store.get_shield.return_value = shield

    # Mock Guardrails API response with status field (legacy format)
    mock_guardrails_post.return_value = {"status": "blocked", "rails_status": {"reason": "harmful_content"}}

    messages = [
        OpenAIUserMessageParam(content="Hello, how are you?"),
        OpenAIAssistantMessageParam(
            content="I'm doing well, thank you for asking!",
            tool_calls=[],
        ),
    ]
    result = await adapter.run_shield(shield_id, messages)

    # Verify the shield store was called
    adapter.shield_store.get_shield.assert_called_once_with(shield_id)

    # Verify the Guardrails API was called correctly
    mock_guardrails_post.assert_called_once_with(
        path="/v1/guardrail/chat/completions",
        data={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
            ],
            "guardrails": {
                "config_id": "self-check",
            },
            "temperature": 1.0,
        },
    )

    # Verify the result
    assert result.violation is not None
    assert isinstance(result, RunShieldResponse)
    assert result.violation.user_message == "Content blocked by guardrails"
    assert result.violation.violation_level == ViolationLevel.ERROR
    assert result.violation.metadata == {"reason": "harmful_content"}


async def test_run_shield_blocked_by_message_match(nvidia_adapter, mock_guardrails_post):
    """Test that shield correctly detects blocks via blocked_message matching."""
    adapter = nvidia_adapter

    shield_id = "test-shield"
    shield = Shield(
        provider_id="nvidia",
        type=ResourceType.shield,
        identifier=shield_id,
        provider_resource_id="test-model",
    )
    adapter.shield_store.get_shield.return_value = shield

    # Mock Guardrails API response with blocked message in choices format (OpenAI compatible)
    mock_guardrails_post.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "I'm sorry, I can't respond to that.",
                }
            }
        ]
    }

    messages = [
        OpenAIUserMessageParam(content="Tell me something harmful"),
    ]
    result = await adapter.run_shield(shield_id, messages)

    adapter.shield_store.get_shield.assert_called_once_with(shield_id)

    mock_guardrails_post.assert_called_once_with(
        path="/v1/guardrail/chat/completions",
        data={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Tell me something harmful"},
            ],
            "guardrails": {
                "config_id": "self-check",
            },
            "temperature": 1.0,
        },
    )

    # Verify the result - should be a violation due to blocked_message match
    assert result.violation is not None
    assert isinstance(result, RunShieldResponse)
    assert result.violation.user_message == "I'm sorry, I can't respond to that."
    assert result.violation.violation_level == ViolationLevel.ERROR
    assert result.violation.metadata == {"matched_pattern": "I'm sorry, I can't respond to that."}


async def test_run_shield_not_found(nvidia_adapter, mock_guardrails_post):
    adapter = nvidia_adapter

    # Set up shield store to return None
    shield_id = "non-existent-shield"
    adapter.shield_store.get_shield.return_value = None

    messages = [
        OpenAIUserMessageParam(content="Hello, how are you?"),
    ]

    with pytest.raises(ValueError):
        await adapter.run_shield(shield_id, messages)

    # Verify the shield store was called
    adapter.shield_store.get_shield.assert_called_once_with(shield_id)

    # Verify the Guardrails API was not called
    mock_guardrails_post.assert_not_called()


async def test_run_shield_http_error(nvidia_adapter, mock_guardrails_post):
    adapter = nvidia_adapter

    shield_id = "test-shield"
    shield = Shield(
        provider_id="nvidia",
        type=ResourceType.shield,
        identifier=shield_id,
        provider_resource_id="test-model",
    )
    adapter.shield_store.get_shield.return_value = shield

    # Mock Guardrails API to raise an exception
    error_msg = "API Error: 500 Internal Server Error"
    mock_guardrails_post.side_effect = Exception(error_msg)

    # Running the shield should raise an exception
    messages = [
        OpenAIUserMessageParam(content="Hello, how are you?"),
        OpenAIAssistantMessageParam(
            content="I'm doing well, thank you for asking!",
            tool_calls=[],
        ),
    ]
    with pytest.raises(Exception) as exc_info:
        await adapter.run_shield(shield_id, messages)

    # Verify the shield store was called
    adapter.shield_store.get_shield.assert_called_once_with(shield_id)

    # Verify the Guardrails API was called correctly
    mock_guardrails_post.assert_called_once_with(
        path="/v1/guardrail/chat/completions",
        data={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
            ],
            "guardrails": {
                "config_id": "self-check",
            },
            "temperature": 1.0,
        },
    )
    # Verify the exception message
    assert error_msg in str(exc_info.value)


def test_init_nemo_guardrails():
    from llama_stack.providers.remote.safety.nvidia.nvidia import NeMoGuardrails

    os.environ["NVIDIA_GUARDRAILS_URL"] = "http://nemo.test"

    test_config_id = "test-custom-config-id"
    config = NVIDIASafetyConfig(
        guardrails_service_url=os.environ["NVIDIA_GUARDRAILS_URL"],
        config_id=test_config_id,
    )
    test_model = "test-model"
    guardrails = NeMoGuardrails(config, test_model)

    # Verify the attributes are set correctly
    assert guardrails.config_id == test_config_id
    assert guardrails.model == test_model
    assert guardrails.guardrails_service_url == os.environ["NVIDIA_GUARDRAILS_URL"]


def test_init_nemo_guardrails_missing_config_id():
    from llama_stack.providers.remote.safety.nvidia.nvidia import NeMoGuardrails

    os.environ["NVIDIA_GUARDRAILS_URL"] = "http://nemo.test"

    config = NVIDIASafetyConfig(
        guardrails_service_url=os.environ["NVIDIA_GUARDRAILS_URL"],
        config_id=None,
    )
    with pytest.raises(ValueError, match="Must provide config_id"):
        NeMoGuardrails(config, "test-model")


# Microservice API mode tests


async def test_run_shield_microservice_allowed(nvidia_adapter_microservice, mock_guardrails_post):
    """Test microservice mode with allowed response."""
    adapter = nvidia_adapter_microservice

    shield_id = "test-shield"
    shield = Shield(
        provider_id="nvidia",
        type=ResourceType.shield,
        identifier=shield_id,
        provider_resource_id="test-model",
    )
    adapter.shield_store.get_shield.return_value = shield

    mock_guardrails_post.return_value = {"status": "allowed"}

    messages = [
        OpenAIUserMessageParam(content="Hello, how are you?"),
        OpenAIAssistantMessageParam(content="I'm doing well!", tool_calls=[]),
    ]
    result = await adapter.run_shield(shield_id, messages)

    adapter.shield_store.get_shield.assert_called_once_with(shield_id)

    mock_guardrails_post.assert_called_once_with(
        path="/v1/guardrail/checks",
        data={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well!"},
            ],
            "temperature": 1.0,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "max_tokens": 160,
            "stream": False,
            "guardrails": {"config_id": "self-check"},
        },
    )

    assert isinstance(result, RunShieldResponse)
    assert result.violation is None


async def test_run_shield_microservice_blocked(nvidia_adapter_microservice, mock_guardrails_post):
    """Test microservice mode with blocked response."""
    adapter = nvidia_adapter_microservice

    shield_id = "test-shield"
    shield = Shield(
        provider_id="nvidia",
        type=ResourceType.shield,
        identifier=shield_id,
        provider_resource_id="test-model",
    )
    adapter.shield_store.get_shield.return_value = shield

    mock_guardrails_post.return_value = {
        "status": "blocked",
        "rails_status": {"self check input": "blocked"},
    }

    messages = [OpenAIUserMessageParam(content="Something harmful")]
    result = await adapter.run_shield(shield_id, messages)

    adapter.shield_store.get_shield.assert_called_once_with(shield_id)

    mock_guardrails_post.assert_called_once_with(
        path="/v1/guardrail/checks",
        data={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Something harmful"}],
            "temperature": 1.0,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "max_tokens": 160,
            "stream": False,
            "guardrails": {"config_id": "self-check"},
        },
    )

    assert result.violation is not None
    assert isinstance(result, RunShieldResponse)
    assert result.violation.user_message == "Sorry I cannot do this."
    assert result.violation.violation_level == ViolationLevel.ERROR
    assert result.violation.metadata == {"self check input": "blocked"}

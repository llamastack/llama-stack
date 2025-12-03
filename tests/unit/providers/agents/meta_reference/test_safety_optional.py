# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Tests for making Safety API optional in meta-reference agents provider.

This test suite validates the changes introduced to fix issue #4165, which
allows running the meta-reference agents provider without the Safety API.
Safety API is now an optional dependency, and errors are raised at request time
when guardrails are explicitly requested without Safety API configured.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llama_stack.core.datatypes import Api
from llama_stack.core.storage.datatypes import KVStoreReference, ResponsesStoreReference
from llama_stack.providers.inline.agents.meta_reference import get_provider_impl
from llama_stack.providers.inline.agents.meta_reference.config import (
    AgentPersistenceConfig,
    MetaReferenceAgentsImplConfig,
)
from llama_stack.providers.inline.agents.meta_reference.responses.utils import (
    run_guardrails,
)


@pytest.fixture
def mock_persistence_config():
    """Create a mock persistence configuration."""
    return AgentPersistenceConfig(
        agent_state=KVStoreReference(
            backend="kv_default",
            namespace="agents",
        ),
        responses=ResponsesStoreReference(
            backend="sql_default",
            table_name="responses",
        ),
    )


@pytest.fixture
def mock_deps():
    """Create mock dependencies for the agents provider."""
    # Create mock APIs
    inference_api = AsyncMock()
    vector_io_api = AsyncMock()
    tool_runtime_api = AsyncMock()
    tool_groups_api = AsyncMock()
    conversations_api = AsyncMock()
    prompts_api = AsyncMock()
    files_api = AsyncMock()

    return {
        Api.inference: inference_api,
        Api.vector_io: vector_io_api,
        Api.tool_runtime: tool_runtime_api,
        Api.tool_groups: tool_groups_api,
        Api.conversations: conversations_api,
        Api.prompts: prompts_api,
        Api.files: files_api,
    }


class TestProviderInitialization:
    """Test provider initialization with different safety API configurations."""

    async def test_initialization_with_safety_api_present(self, mock_persistence_config, mock_deps):
        """Test successful initialization when Safety API is configured."""
        config = MetaReferenceAgentsImplConfig(persistence=mock_persistence_config)

        # Add safety API to deps
        safety_api = AsyncMock()
        mock_deps[Api.safety] = safety_api

        # Mock the initialize method to avoid actual initialization
        with patch(
            "llama_stack.providers.inline.agents.meta_reference.agents.MetaReferenceAgentsImpl.initialize",
            new_callable=AsyncMock,
        ):
            # Should not raise any exception
            provider = await get_provider_impl(config, mock_deps, policy=[])
            assert provider is not None

    async def test_initialization_without_safety_api(self, mock_persistence_config, mock_deps):
        """Test successful initialization when Safety API is not configured."""
        config = MetaReferenceAgentsImplConfig(persistence=mock_persistence_config)

        # Safety API is NOT in mock_deps - provider should still start
        # Mock the initialize method to avoid actual initialization
        with patch(
            "llama_stack.providers.inline.agents.meta_reference.agents.MetaReferenceAgentsImpl.initialize",
            new_callable=AsyncMock,
        ):
            # Should not raise any exception
            provider = await get_provider_impl(config, mock_deps, policy=[])
            assert provider is not None
            assert provider.safety_api is None


class TestGuardrailsFunctionality:
    """Test run_guardrails function with optional safety API."""

    async def test_run_guardrails_with_none_safety_api(self):
        """Test that run_guardrails returns None when safety_api is None."""
        result = await run_guardrails(safety_api=None, messages="test message", guardrail_ids=["llama-guard"])
        assert result is None

    async def test_run_guardrails_with_empty_messages(self):
        """Test that run_guardrails returns None for empty messages."""
        # Test with None safety API
        result = await run_guardrails(safety_api=None, messages="", guardrail_ids=["llama-guard"])
        assert result is None

        # Test with mock safety API
        mock_safety_api = AsyncMock()
        result = await run_guardrails(safety_api=mock_safety_api, messages="", guardrail_ids=["llama-guard"])
        assert result is None

    async def test_run_guardrails_with_none_safety_api_ignores_guardrails(self):
        """Test that guardrails are skipped when safety_api is None, even if guardrail_ids are provided."""
        # Should not raise exception, just return None
        result = await run_guardrails(
            safety_api=None,
            messages="potentially harmful content",
            guardrail_ids=["llama-guard", "content-filter"],
        )
        assert result is None

    async def test_create_response_rejects_guardrails_without_safety_api(self, mock_persistence_config, mock_deps):
        """Test that create_openai_response raises error when guardrails requested but Safety API unavailable."""
        from llama_stack.providers.inline.agents.meta_reference.responses.openai_responses import (
            OpenAIResponsesImpl,
        )
        from llama_stack_api import ResponseGuardrailSpec

        # Create OpenAIResponsesImpl with no safety API
        with patch("llama_stack.providers.inline.agents.meta_reference.responses.openai_responses.ResponsesStore"):
            impl = OpenAIResponsesImpl(
                inference_api=mock_deps[Api.inference],
                tool_groups_api=mock_deps[Api.tool_groups],
                tool_runtime_api=mock_deps[Api.tool_runtime],
                responses_store=MagicMock(),
                vector_io_api=mock_deps[Api.vector_io],
                safety_api=None,  # No Safety API
                conversations_api=mock_deps[Api.conversations],
                prompts_api=mock_deps[Api.prompts],
                files_api=mock_deps[Api.files],
            )

            # Test with string guardrail
            with pytest.raises(ValueError) as exc_info:
                await impl.create_openai_response(
                    input="test input",
                    model="test-model",
                    guardrails=["llama-guard"],
                )
            assert "Cannot process guardrails: Safety API is not configured" in str(exc_info.value)

            # Test with ResponseGuardrailSpec
            with pytest.raises(ValueError) as exc_info:
                await impl.create_openai_response(
                    input="test input",
                    model="test-model",
                    guardrails=[ResponseGuardrailSpec(type="llama-guard")],
                )
            assert "Cannot process guardrails: Safety API is not configured" in str(exc_info.value)

    async def test_create_response_succeeds_without_guardrails_and_no_safety_api(
        self, mock_persistence_config, mock_deps
    ):
        """Test that create_openai_response works when no guardrails requested and Safety API unavailable."""
        from llama_stack.providers.inline.agents.meta_reference.responses.openai_responses import (
            OpenAIResponsesImpl,
        )

        # Create OpenAIResponsesImpl with no safety API
        with (
            patch("llama_stack.providers.inline.agents.meta_reference.responses.openai_responses.ResponsesStore"),
            patch.object(OpenAIResponsesImpl, "_create_streaming_response", new_callable=AsyncMock) as mock_stream,
        ):
            # Mock the streaming response to return a simple async generator
            async def mock_generator():
                yield MagicMock()

            mock_stream.return_value = mock_generator()

            impl = OpenAIResponsesImpl(
                inference_api=mock_deps[Api.inference],
                tool_groups_api=mock_deps[Api.tool_groups],
                tool_runtime_api=mock_deps[Api.tool_runtime],
                responses_store=MagicMock(),
                vector_io_api=mock_deps[Api.vector_io],
                safety_api=None,  # No Safety API
                conversations_api=mock_deps[Api.conversations],
                prompts_api=mock_deps[Api.prompts],
                files_api=mock_deps[Api.files],
            )

            # Should not raise when no guardrails requested
            # Note: This will still fail later in execution due to mocking, but should pass the validation
            try:
                await impl.create_openai_response(
                    input="test input",
                    model="test-model",
                    guardrails=None,  # No guardrails
                )
            except Exception as e:
                # Ensure the error is NOT about missing Safety API
                assert "Cannot process guardrails: Safety API is not configured" not in str(e)

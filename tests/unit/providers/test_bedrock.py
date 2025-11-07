# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from types import SimpleNamespace
from unittest.mock import AsyncMock, PropertyMock, patch

from llama_stack.apis.inference import OpenAIChatCompletionRequestWithExtraBody
from llama_stack.providers.remote.inference.bedrock.bedrock import BedrockInferenceAdapter
from llama_stack.providers.remote.inference.bedrock.config import BedrockConfig


def test_can_create_adapter():
    config = BedrockConfig(api_key="test-key", region_name="us-east-1")
    adapter = BedrockInferenceAdapter(config=config)

    assert adapter is not None
    assert adapter.config.region_name == "us-east-1"
    assert adapter.get_api_key() == "test-key"


def test_different_aws_regions():
    # just check a couple regions to verify URL construction works
    config = BedrockConfig(api_key="key", region_name="us-east-1")
    adapter = BedrockInferenceAdapter(config=config)
    assert adapter.get_base_url() == "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1"

    config = BedrockConfig(api_key="key", region_name="eu-west-1")
    adapter = BedrockInferenceAdapter(config=config)
    assert adapter.get_base_url() == "https://bedrock-runtime.eu-west-1.amazonaws.com/openai/v1"


async def test_basic_chat_completion():
    """Test basic chat completion works with OpenAIMixin"""
    config = BedrockConfig(api_key="k", region_name="us-east-1")
    adapter = BedrockInferenceAdapter(config=config)

    class FakeModelStore:
        async def has_model(self, model_id):
            return True

        async def get_model(self, model_id):
            return SimpleNamespace(provider_resource_id="meta.llama3-1-8b-instruct-v1:0")

    adapter.model_store = FakeModelStore()

    fake_response = SimpleNamespace(
        id="chatcmpl-123",
        choices=[SimpleNamespace(message=SimpleNamespace(content="Hello!", role="assistant"), finish_reason="stop")],
    )

    mock_create = AsyncMock(return_value=fake_response)

    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=mock_create))

    with patch.object(type(adapter), "client", new_callable=PropertyMock, return_value=FakeClient()):
        params = OpenAIChatCompletionRequestWithExtraBody(
            model="llama3-1-8b",
            messages=[{"role": "user", "content": "hello"}],
            stream=False,
        )
        response = await adapter.openai_chat_completion(params=params)

        assert response.id == "chatcmpl-123"
        assert mock_create.await_count == 1

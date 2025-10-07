# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from unittest.mock import MagicMock

import pytest

from llama_stack.core.request_headers import request_provider_data_context
from llama_stack.providers.remote.inference.groq.config import GroqConfig
from llama_stack.providers.remote.inference.groq.groq import GroqInferenceAdapter
from llama_stack.providers.remote.inference.llama_openai_compat.config import LlamaCompatConfig
from llama_stack.providers.remote.inference.llama_openai_compat.llama import LlamaCompatInferenceAdapter
from llama_stack.providers.remote.inference.openai.config import OpenAIConfig
from llama_stack.providers.remote.inference.openai.openai import OpenAIInferenceAdapter
from llama_stack.providers.remote.inference.together.config import TogetherImplConfig
from llama_stack.providers.remote.inference.together.together import TogetherInferenceAdapter
from llama_stack.providers.remote.inference.watsonx.config import WatsonXConfig
from llama_stack.providers.remote.inference.watsonx.watsonx import WatsonXInferenceAdapter


@pytest.mark.parametrize(
    "config_cls,adapter_cls,provider_data_validator",
    [
        (
            GroqConfig,
            GroqInferenceAdapter,
            "llama_stack.providers.remote.inference.groq.config.GroqProviderDataValidator",
        ),
        (
            OpenAIConfig,
            OpenAIInferenceAdapter,
            "llama_stack.providers.remote.inference.openai.config.OpenAIProviderDataValidator",
        ),
        (
            TogetherImplConfig,
            TogetherInferenceAdapter,
            "llama_stack.providers.remote.inference.together.TogetherProviderDataValidator",
        ),
        (
            LlamaCompatConfig,
            LlamaCompatInferenceAdapter,
            "llama_stack.providers.remote.inference.llama_openai_compat.config.LlamaProviderDataValidator",
        ),
    ],
)
def test_openai_provider_data_used(config_cls, adapter_cls, provider_data_validator: str):
    """Ensure the OpenAI provider does not cache api keys across client requests"""

    inference_adapter = adapter_cls(config=config_cls())

    inference_adapter.__provider_spec__ = MagicMock()
    inference_adapter.__provider_spec__.provider_data_validator = provider_data_validator

    for api_key in ["test1", "test2"]:
        with request_provider_data_context(
            {"x-llamastack-provider-data": json.dumps({inference_adapter.provider_data_api_key_field: api_key})}
        ):
            assert inference_adapter.client.api_key == api_key


@pytest.mark.parametrize(
    "config_cls,adapter_cls,provider_data_validator",
    [
        (
            WatsonXConfig,
            WatsonXInferenceAdapter,
            "llama_stack.providers.remote.inference.watsonx.config.WatsonXProviderDataValidator",
        ),
    ],
)
def test_litellm_provider_data_used(config_cls, adapter_cls, provider_data_validator: str):
    """Validate data for LiteLLM-based providers.  Similar to test_openai_provider_data_used, but without the
    assumption that there is an OpenAI-compatible client object."""

    inference_adapter = adapter_cls(config=config_cls())

    inference_adapter.__provider_spec__ = MagicMock()
    inference_adapter.__provider_spec__.provider_data_validator = provider_data_validator

    for api_key in ["test1", "test2"]:
        with request_provider_data_context(
            {"x-llamastack-provider-data": json.dumps({inference_adapter.provider_data_api_key_field: api_key})}
        ):
            assert inference_adapter.get_api_key() == api_key

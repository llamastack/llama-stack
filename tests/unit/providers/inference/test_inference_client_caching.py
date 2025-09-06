# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from unittest.mock import MagicMock

from llama_stack.core.request_headers import request_provider_data_context
from llama_stack.providers.remote.inference.groq.config import GroqConfig
from llama_stack.providers.remote.inference.groq.groq import GroqInferenceAdapter
from llama_stack.providers.remote.inference.llama_openai_compat.config import LlamaCompatConfig
from llama_stack.providers.remote.inference.llama_openai_compat.llama import LlamaCompatInferenceAdapter
from llama_stack.providers.remote.inference.openai.config import OpenAIConfig
from llama_stack.providers.remote.inference.openai.openai import OpenAIInferenceAdapter
from llama_stack.providers.remote.inference.together.config import TogetherImplConfig
from llama_stack.providers.remote.inference.together.together import TogetherInferenceAdapter


def test_groq_provider_openai_client_caching():
    """Ensure the Groq provider does not cache api keys across client requests"""

    config = GroqConfig()
    inference_adapter = GroqInferenceAdapter(config)

    inference_adapter.__provider_spec__ = MagicMock()
    inference_adapter.__provider_spec__.provider_data_validator = (
        "llama_stack.providers.remote.inference.groq.config.GroqProviderDataValidator"
    )

    for api_key in ["test1", "test2"]:
        with request_provider_data_context(
            {"x-llamastack-provider-data": json.dumps({inference_adapter.provider_data_api_key_field: api_key})}
        ):
            assert inference_adapter.client.api_key == api_key


def test_openai_provider_openai_client_caching():
    """Ensure the OpenAI provider does not cache api keys across client requests"""

    config = OpenAIConfig()
    inference_adapter = OpenAIInferenceAdapter(config)

    inference_adapter.__provider_spec__ = MagicMock()
    inference_adapter.__provider_spec__.provider_data_validator = (
        "llama_stack.providers.remote.inference.openai.config.OpenAIProviderDataValidator"
    )

    for api_key in ["test1", "test2"]:
        with request_provider_data_context(
            {"x-llamastack-provider-data": json.dumps({inference_adapter.provider_data_api_key_field: api_key})}
        ):
            openai_client = inference_adapter.client
            assert openai_client.api_key == api_key


def test_together_provider_openai_client_caching():
    """Ensure the Together provider does not cache api keys across client requests"""

    config = TogetherImplConfig()
    inference_adapter = TogetherInferenceAdapter(config)

    inference_adapter.__provider_spec__ = MagicMock()
    inference_adapter.__provider_spec__.provider_data_validator = (
        "llama_stack.providers.remote.inference.together.TogetherProviderDataValidator"
    )

    for api_key in ["test1", "test2"]:
        with request_provider_data_context({"x-llamastack-provider-data": json.dumps({"together_api_key": api_key})}):
            together_client = inference_adapter._get_client()
            assert together_client.client.api_key == api_key
            openai_client = inference_adapter._get_openai_client()
            assert openai_client.api_key == api_key


def test_llama_compat_provider_openai_client_caching():
    """Ensure the LlamaCompat provider does not cache api keys across client requests"""
    config = LlamaCompatConfig()
    inference_adapter = LlamaCompatInferenceAdapter(config)

    inference_adapter.__provider_spec__ = MagicMock()
    inference_adapter.__provider_spec__.provider_data_validator = (
        "llama_stack.providers.remote.inference.llama_openai_compat.config.LlamaProviderDataValidator"
    )

    for api_key in ["test1", "test2"]:
        with request_provider_data_context({"x-llamastack-provider-data": json.dumps({"llama_api_key": api_key})}):
            assert inference_adapter.client.api_key == api_key

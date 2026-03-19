# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from unittest.mock import MagicMock

import pytest

from llama_stack.core.request_headers import request_provider_data_context
from llama_stack.providers.remote.inference.anthropic.anthropic import AnthropicInferenceAdapter
from llama_stack.providers.remote.inference.anthropic.config import AnthropicConfig
from llama_stack.providers.remote.inference.cerebras.cerebras import CerebrasInferenceAdapter
from llama_stack.providers.remote.inference.cerebras.config import CerebrasImplConfig
from llama_stack.providers.remote.inference.databricks.config import DatabricksImplConfig
from llama_stack.providers.remote.inference.databricks.databricks import DatabricksInferenceAdapter
from llama_stack.providers.remote.inference.fireworks.config import FireworksImplConfig
from llama_stack.providers.remote.inference.fireworks.fireworks import FireworksInferenceAdapter
from llama_stack.providers.remote.inference.gemini.config import GeminiConfig
from llama_stack.providers.remote.inference.gemini.gemini import GeminiInferenceAdapter
from llama_stack.providers.remote.inference.groq.config import GroqConfig
from llama_stack.providers.remote.inference.groq.groq import GroqInferenceAdapter
from llama_stack.providers.remote.inference.llama_openai_compat.config import LlamaCompatConfig
from llama_stack.providers.remote.inference.llama_openai_compat.llama import LlamaCompatInferenceAdapter
from llama_stack.providers.remote.inference.nvidia.config import NVIDIAConfig
from llama_stack.providers.remote.inference.nvidia.nvidia import NVIDIAInferenceAdapter
from llama_stack.providers.remote.inference.openai.config import OpenAIConfig
from llama_stack.providers.remote.inference.openai.openai import OpenAIInferenceAdapter
from llama_stack.providers.remote.inference.runpod.config import RunpodImplConfig
from llama_stack.providers.remote.inference.runpod.runpod import RunpodInferenceAdapter
from llama_stack.providers.remote.inference.sambanova.config import SambaNovaImplConfig
from llama_stack.providers.remote.inference.sambanova.sambanova import SambaNovaInferenceAdapter
from llama_stack.providers.remote.inference.together.config import TogetherImplConfig
from llama_stack.providers.remote.inference.together.together import TogetherInferenceAdapter
from llama_stack.providers.remote.inference.vllm.config import VLLMInferenceAdapterConfig
from llama_stack.providers.remote.inference.vllm.vllm import VLLMInferenceAdapter
from llama_stack.providers.remote.inference.watsonx.config import WatsonXConfig
from llama_stack.providers.remote.inference.watsonx.watsonx import WatsonXInferenceAdapter


@pytest.mark.parametrize(
    "config_cls,adapter_cls,provider_data_validator,config_params",
    [
        (
            GroqConfig,
            GroqInferenceAdapter,
            "llama_stack.providers.remote.inference.groq.config.GroqProviderDataValidator",
            {},
        ),
        (
            OpenAIConfig,
            OpenAIInferenceAdapter,
            "llama_stack.providers.remote.inference.openai.config.OpenAIProviderDataValidator",
            {},
        ),
        (
            TogetherImplConfig,
            TogetherInferenceAdapter,
            "llama_stack.providers.remote.inference.together.TogetherProviderDataValidator",
            {},
        ),
        (
            LlamaCompatConfig,
            LlamaCompatInferenceAdapter,
            "llama_stack.providers.remote.inference.llama_openai_compat.config.LlamaProviderDataValidator",
            {},
        ),
        (
            CerebrasImplConfig,
            CerebrasInferenceAdapter,
            "llama_stack.providers.remote.inference.cerebras.config.CerebrasProviderDataValidator",
            {},
        ),
        (
            DatabricksImplConfig,
            DatabricksInferenceAdapter,
            "llama_stack.providers.remote.inference.databricks.config.DatabricksProviderDataValidator",
            {},
        ),
        (
            NVIDIAConfig,
            NVIDIAInferenceAdapter,
            "llama_stack.providers.remote.inference.nvidia.config.NVIDIAProviderDataValidator",
            {},
        ),
        (
            RunpodImplConfig,
            RunpodInferenceAdapter,
            "llama_stack.providers.remote.inference.runpod.config.RunpodProviderDataValidator",
            {},
        ),
        (
            FireworksImplConfig,
            FireworksInferenceAdapter,
            "llama_stack.providers.remote.inference.fireworks.FireworksProviderDataValidator",
            {},
        ),
        (
            AnthropicConfig,
            AnthropicInferenceAdapter,
            "llama_stack.providers.remote.inference.anthropic.config.AnthropicProviderDataValidator",
            {},
        ),
        (
            GeminiConfig,
            GeminiInferenceAdapter,
            "llama_stack.providers.remote.inference.gemini.config.GeminiProviderDataValidator",
            {},
        ),
        (
            SambaNovaImplConfig,
            SambaNovaInferenceAdapter,
            "llama_stack.providers.remote.inference.sambanova.config.SambaNovaProviderDataValidator",
            {},
        ),
        (
            VLLMInferenceAdapterConfig,
            VLLMInferenceAdapter,
            "llama_stack.providers.remote.inference.vllm.VLLMProviderDataValidator",
            {
                "base_url": "http://fake",
            },
        ),
    ],
)
def test_openai_provider_data_used(config_cls, adapter_cls, provider_data_validator: str, config_params: dict):
    """Ensure the OpenAI provider does not cache api keys across client requests"""
    inference_adapter = adapter_cls(config=config_cls(**config_params))

    inference_adapter.__provider_spec__ = MagicMock()
    inference_adapter.__provider_spec__.provider_data_validator = provider_data_validator

    for api_key in ["test1", "test2"]:
        with request_provider_data_context(
            {"x-llamastack-provider-data": json.dumps({inference_adapter.provider_data_api_key_field: api_key})}
        ):
            assert inference_adapter.client.api_key == api_key


@pytest.mark.parametrize(
    "config_cls,adapter_cls,provider_data_validator,config_params,extra_provider_data",
    [
        (
            VLLMInferenceAdapterConfig,
            VLLMInferenceAdapter,
            "llama_stack.providers.remote.inference.vllm.VLLMProviderDataValidator",
            {"base_url": "http://fake"},
            {},
        ),
        (
            OpenAIConfig,
            OpenAIInferenceAdapter,
            "llama_stack.providers.remote.inference.openai.config.OpenAIProviderDataValidator",
            {},
            {"openai_api_key": "test-key"},
        ),
    ],
)
def test_extra_headers_from_provider_data(
    config_cls, adapter_cls, provider_data_validator: str, config_params: dict, extra_provider_data: dict
):
    """Ensure extra headers from provider data are forwarded to the OpenAI client."""
    inference_adapter = adapter_cls(config=config_cls(**config_params))

    inference_adapter.__provider_spec__ = MagicMock()
    inference_adapter.__provider_spec__.provider_data_validator = provider_data_validator

    extra_headers = {"X-Custom-Header": "custom-value", "X-Another": "another-value"}
    provider_data = {inference_adapter.provider_data_extra_headers_field: extra_headers, **extra_provider_data}

    with request_provider_data_context({"x-llamastack-provider-data": json.dumps(provider_data)}):
        assert inference_adapter._get_provider_data_extra_headers() == extra_headers
        assert inference_adapter.get_extra_client_params() == {"default_headers": extra_headers}
        assert extra_headers.items() <= dict(inference_adapter.client.default_headers).items()


def test_extra_headers_absent_returns_empty():
    """Ensure get_extra_client_params returns empty dict when no extra headers are set."""
    inference_adapter = VLLMInferenceAdapter(config=VLLMInferenceAdapterConfig(base_url="http://fake"))

    inference_adapter.__provider_spec__ = MagicMock()
    inference_adapter.__provider_spec__.provider_data_validator = (
        "llama_stack.providers.remote.inference.vllm.VLLMProviderDataValidator"
    )

    with request_provider_data_context({"x-llamastack-provider-data": json.dumps({})}):
        assert inference_adapter._get_provider_data_extra_headers() == {}
        assert inference_adapter.get_extra_client_params() == {}


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
def test_watsonx_provider_data_used(config_cls, adapter_cls, provider_data_validator: str):
    """Validate that WatsonX picks up API key from provider data headers."""

    inference_adapter = adapter_cls(config=config_cls(base_url="http://fake"))

    inference_adapter.__provider_spec__ = MagicMock()
    inference_adapter.__provider_spec__.provider_data_validator = provider_data_validator

    for api_key in ["test1", "test2"]:
        with request_provider_data_context(
            {"x-llamastack-provider-data": json.dumps({inference_adapter.provider_data_api_key_field: api_key})}
        ):
            assert inference_adapter._get_api_key_from_config_or_provider_data() == api_key

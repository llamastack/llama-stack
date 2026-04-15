# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from unittest.mock import MagicMock

import pytest
from llama_stack.core.request_headers import request_provider_data_context
from llama_stack_provider_inference_anthropic.anthropic import AnthropicInferenceAdapter
from llama_stack_provider_inference_anthropic.config import AnthropicConfig
from llama_stack_provider_inference_cerebras.cerebras import CerebrasInferenceAdapter
from llama_stack_provider_inference_cerebras.config import CerebrasImplConfig
from llama_stack_provider_inference_databricks.config import DatabricksImplConfig
from llama_stack_provider_inference_databricks.databricks import DatabricksInferenceAdapter
from llama_stack_provider_inference_fireworks.config import FireworksImplConfig
from llama_stack_provider_inference_fireworks.fireworks import FireworksInferenceAdapter
from llama_stack_provider_inference_gemini.config import GeminiConfig
from llama_stack_provider_inference_gemini.gemini import GeminiInferenceAdapter
from llama_stack_provider_inference_groq.config import GroqConfig
from llama_stack_provider_inference_groq.groq import GroqInferenceAdapter
from llama_stack_provider_inference_llama_openai_compat.config import LlamaCompatConfig
from llama_stack_provider_inference_llama_openai_compat.llama import LlamaCompatInferenceAdapter
from llama_stack_provider_inference_nvidia.config import NVIDIAConfig
from llama_stack_provider_inference_nvidia.nvidia import NVIDIAInferenceAdapter
from llama_stack_provider_inference_openai.config import OpenAIConfig
from llama_stack_provider_inference_openai.openai import OpenAIInferenceAdapter
from llama_stack_provider_inference_runpod.config import RunpodImplConfig
from llama_stack_provider_inference_runpod.runpod import RunpodInferenceAdapter
from llama_stack_provider_inference_sambanova.config import SambaNovaImplConfig
from llama_stack_provider_inference_sambanova.sambanova import SambaNovaInferenceAdapter
from llama_stack_provider_inference_together.config import TogetherImplConfig
from llama_stack_provider_inference_together.together import TogetherInferenceAdapter
from llama_stack_provider_inference_vllm.config import VLLMInferenceAdapterConfig
from llama_stack_provider_inference_vllm.vllm import VLLMInferenceAdapter
from llama_stack_provider_inference_watsonx.config import WatsonXConfig
from llama_stack_provider_inference_watsonx.watsonx import WatsonXInferenceAdapter


@pytest.mark.parametrize(
    "config_cls,adapter_cls,provider_data_validator,config_params",
    [
        (
            GroqConfig,
            GroqInferenceAdapter,
            "llama_stack_provider_inference_groq.config.GroqProviderDataValidator",
            {},
        ),
        (
            OpenAIConfig,
            OpenAIInferenceAdapter,
            "llama_stack_provider_inference_openai.config.OpenAIProviderDataValidator",
            {},
        ),
        (
            TogetherImplConfig,
            TogetherInferenceAdapter,
            "llama_stack_provider_inference_together.TogetherProviderDataValidator",
            {},
        ),
        (
            LlamaCompatConfig,
            LlamaCompatInferenceAdapter,
            "llama_stack_provider_inference_llama_openai_compat.config.LlamaProviderDataValidator",
            {},
        ),
        (
            CerebrasImplConfig,
            CerebrasInferenceAdapter,
            "llama_stack_provider_inference_cerebras.config.CerebrasProviderDataValidator",
            {},
        ),
        (
            DatabricksImplConfig,
            DatabricksInferenceAdapter,
            "llama_stack_provider_inference_databricks.config.DatabricksProviderDataValidator",
            {},
        ),
        (
            NVIDIAConfig,
            NVIDIAInferenceAdapter,
            "llama_stack_provider_inference_nvidia.config.NVIDIAProviderDataValidator",
            {},
        ),
        (
            RunpodImplConfig,
            RunpodInferenceAdapter,
            "llama_stack_provider_inference_runpod.config.RunpodProviderDataValidator",
            {},
        ),
        (
            FireworksImplConfig,
            FireworksInferenceAdapter,
            "llama_stack_provider_inference_fireworks.FireworksProviderDataValidator",
            {},
        ),
        (
            AnthropicConfig,
            AnthropicInferenceAdapter,
            "llama_stack_provider_inference_anthropic.config.AnthropicProviderDataValidator",
            {},
        ),
        (
            GeminiConfig,
            GeminiInferenceAdapter,
            "llama_stack_provider_inference_gemini.config.GeminiProviderDataValidator",
            {},
        ),
        (
            SambaNovaImplConfig,
            SambaNovaInferenceAdapter,
            "llama_stack_provider_inference_sambanova.config.SambaNovaProviderDataValidator",
            {},
        ),
        (
            VLLMInferenceAdapterConfig,
            VLLMInferenceAdapter,
            "llama_stack_provider_inference_vllm.VLLMProviderDataValidator",
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
    "config_cls,adapter_cls,provider_data_validator",
    [
        (
            WatsonXConfig,
            WatsonXInferenceAdapter,
            "llama_stack_provider_inference_watsonx.config.WatsonXProviderDataValidator",
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

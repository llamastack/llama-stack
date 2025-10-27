# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from llama_stack.core.stack import replace_env_vars
from llama_stack.providers.remote.inference.anthropic import AnthropicConfig
from llama_stack.providers.remote.inference.azure import AzureConfig
from llama_stack.providers.remote.inference.bedrock import BedrockConfig
from llama_stack.providers.remote.inference.cerebras import CerebrasImplConfig
from llama_stack.providers.remote.inference.databricks import DatabricksImplConfig
from llama_stack.providers.remote.inference.fireworks import FireworksImplConfig
from llama_stack.providers.remote.inference.gemini import GeminiConfig
from llama_stack.providers.remote.inference.groq import GroqConfig
from llama_stack.providers.remote.inference.llama_openai_compat import LlamaCompatConfig
from llama_stack.providers.remote.inference.nvidia import NVIDIAConfig
from llama_stack.providers.remote.inference.ollama import OllamaImplConfig
from llama_stack.providers.remote.inference.openai import OpenAIConfig
from llama_stack.providers.remote.inference.runpod import RunpodImplConfig
from llama_stack.providers.remote.inference.sambanova import SambaNovaImplConfig
from llama_stack.providers.remote.inference.tgi import TGIImplConfig
from llama_stack.providers.remote.inference.together import TogetherImplConfig
from llama_stack.providers.remote.inference.vertexai import VertexAIConfig
from llama_stack.providers.remote.inference.vllm import VLLMInferenceAdapterConfig
from llama_stack.providers.remote.inference.watsonx import WatsonXConfig


class TestRemoteInferenceProviderConfig:
    @pytest.mark.parametrize(
        "config_cls,alias_name,env_name,extra_config",
        [
            (AnthropicConfig, "api_key", "ANTHROPIC_API_KEY", {}),
            (AzureConfig, "api_key", "AZURE_API_KEY", {"api_base": "HTTP://FAKE"}),
            (BedrockConfig, None, None, {}),
            (CerebrasImplConfig, "api_key", "CEREBRAS_API_KEY", {}),
            (DatabricksImplConfig, "api_token", "DATABRICKS_TOKEN", {}),
            (FireworksImplConfig, "api_key", "FIREWORKS_API_KEY", {}),
            (GeminiConfig, "api_key", "GEMINI_API_KEY", {}),
            (GroqConfig, "api_key", "GROQ_API_KEY", {}),
            (LlamaCompatConfig, "api_key", "LLAMA_API_KEY", {}),
            (NVIDIAConfig, "api_key", "NVIDIA_API_KEY", {}),
            (OllamaImplConfig, None, None, {}),
            (OpenAIConfig, "api_key", "OPENAI_API_KEY", {}),
            (RunpodImplConfig, "api_token", "RUNPOD_API_TOKEN", {}),
            (SambaNovaImplConfig, "api_key", "SAMBANOVA_API_KEY", {}),
            (TGIImplConfig, None, None, {"url": "FAKE"}),
            (TogetherImplConfig, "api_key", "TOGETHER_API_KEY", {}),
            (VertexAIConfig, None, None, {"project": "FAKE", "location": "FAKE"}),
            (VLLMInferenceAdapterConfig, "api_token", "VLLM_API_TOKEN", {}),
            (WatsonXConfig, "api_key", "WATSONX_API_KEY", {}),
        ],
    )
    def test_provider_config_auth_credentials(self, monkeypatch, config_cls, alias_name, env_name, extra_config):
        """Test that the config class correctly maps the alias to auth_credential."""
        secret_value = config_cls.__name__

        if alias_name is None:
            pytest.skip("No alias name provided for this config class.")

        config = config_cls(**{alias_name: secret_value, **extra_config})
        assert config.auth_credential is not None
        assert config.auth_credential.get_secret_value() == secret_value

        schema = config_cls.model_json_schema()
        assert alias_name in schema["properties"]
        assert "auth_credential" not in schema["properties"]

        if env_name:
            monkeypatch.setenv(env_name, secret_value)
            sample_config = config_cls.sample_run_config()
            expanded_config = replace_env_vars(sample_config)
            config_from_sample = config_cls(**{**expanded_config, **extra_config})
            assert config_from_sample.auth_credential is not None
            assert config_from_sample.auth_credential.get_secret_value() == secret_value

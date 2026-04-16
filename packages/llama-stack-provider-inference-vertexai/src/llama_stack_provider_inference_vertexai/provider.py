# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_api import Api, RemoteProviderSpec


def get_provider_spec() -> RemoteProviderSpec:
    return RemoteProviderSpec(
        api=Api.inference,
        adapter_type="vertexai",
        provider_type="remote::vertexai",
        pip_packages=[],
        module="llama_stack_provider_inference_vertexai",
        config_class="llama_stack_provider_inference_vertexai.VertexAIConfig",
        provider_data_validator="llama_stack_provider_inference_vertexai.config.VertexAIProviderDataValidator",
        description="""Google Vertex AI inference provider enables you to use Google's Gemini models through Google Cloud's Vertex AI platform, providing several advantages:

• Enterprise-grade security: Uses Google Cloud's security controls and IAM
• Better integration: Seamless integration with other Google Cloud services
• Advanced features: Access to additional Vertex AI features like model tuning and monitoring
• Authentication: Uses Google Cloud Application Default Credentials (ADC) instead of API keys

Configuration:
- Set VERTEX_AI_PROJECT environment variable (required)
- Set VERTEX_AI_LOCATION environment variable (optional, defaults to us-central1)
- Use Google Cloud Application Default Credentials or service account key

Authentication Setup:
Option 1 (Recommended): gcloud auth application-default login
Option 2: Set GOOGLE_APPLICATION_CREDENTIALS to service account key path

Available Models:
- vertex_ai/gemini-2.0-flash
- vertex_ai/gemini-2.5-flash
- vertex_ai/gemini-2.5-pro""",
    )

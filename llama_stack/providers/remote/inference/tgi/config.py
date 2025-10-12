# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from pydantic import BaseModel, Field, SecretStr

from llama_stack.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from llama_stack.schema_utils import json_schema_type


@json_schema_type
class TGIImplConfig(RemoteInferenceProviderConfig):
    auth_credential: SecretStr | None = Field(default=None, exclude=True)

    url: str = Field(
        description="The URL for the TGI serving endpoint",
    )

    @classmethod
    def sample_run_config(
        cls,
        url: str = "${env.TGI_URL:=}",
        **kwargs,
    ):
        return {
            "url": url,
        }


@json_schema_type
class InferenceEndpointImplConfig(BaseModel):
    endpoint_name: str = Field(
        description="The name of the Hugging Face Inference Endpoint in the format of '{namespace}/{endpoint_name}' (e.g. 'my-cool-org/meta-llama-3-1-8b-instruct-rce'). Namespace is optional and will default to the user account if not provided.",
    )
    api_token: SecretStr | None = Field(
        default=None,
        description="Your Hugging Face user access token (will default to locally saved token if not provided)",
    )

    @classmethod
    def sample_run_config(
        cls,
        endpoint_name: str = "${env.INFERENCE_ENDPOINT_NAME}",
        api_token: str = "${env.HF_API_TOKEN}",
        **kwargs,
    ):
        return {
            "endpoint_name": endpoint_name,
            "api_token": api_token,
        }


@json_schema_type
class InferenceAPIImplConfig(BaseModel):
    huggingface_repo: str = Field(
        description="The model ID of the model on the Hugging Face Hub (e.g. 'meta-llama/Meta-Llama-3.1-70B-Instruct')",
    )
    api_token: SecretStr | None = Field(
        default=None,
        description="Your Hugging Face user access token (will default to locally saved token if not provided)",
    )

    @classmethod
    def sample_run_config(
        cls,
        repo: str = "${env.INFERENCE_MODEL}",
        api_token: str = "${env.HF_API_TOKEN}",
        **kwargs,
    ):
        return {
            "huggingface_repo": repo,
            "api_token": api_token,
        }

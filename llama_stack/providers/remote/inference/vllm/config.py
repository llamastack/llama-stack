# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path

from pydantic import Field, SecretStr, field_validator

from llama_stack.providers.utils.inference.model_registry import RemoteInferenceProviderConfig
from llama_stack.schema_utils import json_schema_type


@json_schema_type
class VLLMInferenceAdapterConfig(RemoteInferenceProviderConfig):
    url: str | None = Field(
        default=None,
        description="The URL for the vLLM model serving endpoint",
    )
    max_tokens: int = Field(
        default=4096,
        description="Maximum number of tokens to generate.",
    )
    auth_credential: SecretStr | None = Field(
        default=None,
        alias="api_token",
        description="The API token",
    )
    tls_verify: bool | str = Field(
        default=True,
        description="Whether to verify TLS certificates. Can be a boolean or a path to a CA certificate file.",
    )

    @field_validator("tls_verify")
    @classmethod
    def validate_tls_verify(cls, v):
        if isinstance(v, str):
            # Otherwise, treat it as a cert path
            cert_path = Path(v).expanduser().resolve()
            if not cert_path.exists():
                raise ValueError(f"TLS certificate file does not exist: {v}")
            if not cert_path.is_file():
                raise ValueError(f"TLS certificate path is not a file: {v}")
            return v
        return v

    @classmethod
    def sample_run_config(
        cls,
        url: str = "${env.VLLM_URL:=}",
        **kwargs,
    ):
        return {
            "url": url,
            "max_tokens": "${env.VLLM_MAX_TOKENS:=4096}",
            "api_token": "${env.VLLM_API_TOKEN:=fake}",
            "tls_verify": "${env.VLLM_TLS_VERIFY:=true}",
        }

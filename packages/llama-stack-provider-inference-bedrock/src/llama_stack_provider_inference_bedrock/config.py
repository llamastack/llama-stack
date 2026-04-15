# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

from llama_stack_utils_inference.model_registry import RemoteInferenceProviderConfig
from pydantic import BaseModel, Field, SecretStr


class BedrockProviderDataValidator(BaseModel):
    """Validates provider-specific request data for AWS Bedrock inference."""

    aws_bearer_token_bedrock: SecretStr | None = Field(
        default=None,
        description="API Key (Bearer token) for Amazon Bedrock",
    )


class BedrockConfig(RemoteInferenceProviderConfig):
    """Configuration for the AWS Bedrock inference provider."""

    region_name: str = Field(
        default_factory=lambda: os.getenv("AWS_DEFAULT_REGION", "us-east-2"),
        description="AWS Region for the Bedrock Runtime endpoint",
    )

    @classmethod
    def sample_run_config(cls, **kwargs):
        return {
            "api_key": "${env.AWS_BEARER_TOKEN_BEDROCK:=}",
            "region_name": "${env.AWS_DEFAULT_REGION:=us-east-2}",
        }

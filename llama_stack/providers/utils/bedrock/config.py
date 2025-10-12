# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

from pydantic import Field

from llama_stack.providers.utils.inference.model_registry import RemoteInferenceProviderConfig


class BedrockBaseConfig(RemoteInferenceProviderConfig):
    auth_credential: None = Field(default=None, exclude=True)
    aws_access_key_id: str | None = Field(
        default_factory=lambda: os.getenv("AWS_ACCESS_KEY_ID"),
        description="The AWS access key to use. Default use environment variable: AWS_ACCESS_KEY_ID",
    )
    aws_secret_access_key: str | None = Field(
        default_factory=lambda: os.getenv("AWS_SECRET_ACCESS_KEY"),
        description="The AWS secret access key to use. Default use environment variable: AWS_SECRET_ACCESS_KEY",
    )
    aws_session_token: str | None = Field(
        default_factory=lambda: os.getenv("AWS_SESSION_TOKEN"),
        description="The AWS session token to use. Default use environment variable: AWS_SESSION_TOKEN",
    )
    region_name: str | None = Field(
        default_factory=lambda: os.getenv("AWS_DEFAULT_REGION"),
        description="The default AWS Region to use, for example, us-west-1 or us-west-2."
        "Default use environment variable: AWS_DEFAULT_REGION",
    )
    profile_name: str | None = Field(
        default_factory=lambda: os.getenv("AWS_PROFILE"),
        description="The profile name that contains credentials to use.Default use environment variable: AWS_PROFILE",
    )
    total_max_attempts: int | None = Field(
        default_factory=lambda: int(val) if (val := os.getenv("AWS_MAX_ATTEMPTS")) else None,
        description="An integer representing the maximum number of attempts that will be made for a single request, "
        "including the initial attempt. Default use environment variable: AWS_MAX_ATTEMPTS",
    )
    retry_mode: str | None = Field(
        default_factory=lambda: os.getenv("AWS_RETRY_MODE"),
        description="A string representing the type of retries Boto3 will perform."
        "Default use environment variable: AWS_RETRY_MODE",
    )
    connect_timeout: float | None = Field(
        default_factory=lambda: float(os.getenv("AWS_CONNECT_TIMEOUT", "60")),
        description="The time in seconds till a timeout exception is thrown when attempting to make a connection. "
        "The default is 60 seconds.",
    )
    read_timeout: float | None = Field(
        default_factory=lambda: float(os.getenv("AWS_READ_TIMEOUT", "60")),
        description="The time in seconds till a timeout exception is thrown when attempting to read from a connection."
        "The default is 60 seconds.",
    )
    session_ttl: int | None = Field(
        default_factory=lambda: int(os.getenv("AWS_SESSION_TTL", "3600")),
        description="The time in seconds till a session expires. The default is 3600 seconds (1 hour).",
    )

    @classmethod
    def sample_run_config(cls, **kwargs):
        return {}

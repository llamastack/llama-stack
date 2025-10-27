# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import boto3
from botocore.client import BaseClient
from botocore.config import Config

from llama_stack.providers.utils.bedrock.config import BedrockBaseConfig
from llama_stack.providers.utils.bedrock.refreshable_boto_session import (
    RefreshableBotoSession,
)


def create_bedrock_client(config: BedrockBaseConfig, service_name: str = "bedrock-runtime") -> BaseClient:
    """Creates a boto3 client for Bedrock services with the given configuration.

    Args:
        config: The Bedrock configuration containing AWS credentials and settings
        service_name: The AWS service name to create client for (default: "bedrock-runtime")

    Returns:
        A configured boto3 client
    """
    if config.aws_access_key_id and config.aws_secret_access_key:
        retries_config = {
            k: v
            for k, v in dict(
                total_max_attempts=config.total_max_attempts,
                mode=config.retry_mode,
            ).items()
            if v is not None
        }

        config_args = {
            k: v
            for k, v in dict(
                region_name=config.region_name,
                retries=retries_config if retries_config else None,
                connect_timeout=config.connect_timeout,
                read_timeout=config.read_timeout,
            ).items()
            if v is not None
        }

        boto3_config = Config(**config_args)

        session_args = {
            "aws_access_key_id": config.aws_access_key_id,
            "aws_secret_access_key": config.aws_secret_access_key,
            "aws_session_token": config.aws_session_token,
            "region_name": config.region_name,
            "profile_name": config.profile_name,
            "session_ttl": config.session_ttl,
        }

        # Remove None values
        session_args = {k: v for k, v in session_args.items() if v is not None}

        boto3_session = boto3.session.Session(**session_args)
        return boto3_session.client(service_name, config=boto3_config)
    else:
        return (
            RefreshableBotoSession(
                region_name=config.region_name,
                profile_name=config.profile_name,
                session_ttl=config.session_ttl,
            )
            .refreshable_session()
            .client(service_name)
        )

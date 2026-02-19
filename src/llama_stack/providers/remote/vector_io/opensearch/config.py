# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Optional

from llama_stack_api.shared.schemas import AdapterSpec, StackComponentConfig
from pydantic import Field, SecretStr


class OpenSearchVectorIOConfig(StackComponentConfig):
    host: str = Field(
        default="localhost",
        description="The host of the OpenSearch server",
    )
    port: int = Field(
        default=9200,
        description="The port of the OpenSearch server",
    )
    use_ssl: bool = Field(
        default=False,
        description="Whether to use SSL for the connection",
    )
    verify_certs: bool = Field(
        default=False,
        description="Whether to verify SSL certificates",
    )
    username: Optional[str] = Field(
        default=None,
        description="The username for authentication",
    )
    password: Optional[SecretStr] = Field(
        default=None,
        description="The password for authentication",
    )

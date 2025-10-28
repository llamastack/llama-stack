# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from llama_stack.core.storage.datatypes import SqlStoreReference


class OpenAIFilesImplConfig(BaseModel):
    """Configuration for OpenAI Files API provider."""

    api_key: str = Field(description="OpenAI API key for authentication")
    api_base: str = Field(default="https://api.openai.com/v1", description="OpenAI API base URL")
    metadata_store: SqlStoreReference = Field(description="SQL store configuration for file metadata")

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {
            "api_key": "${env.OPENAI_API_KEY}",
            "api_base": "${env.OPENAI_API_BASE:=https://api.openai.com/v1}",
            "metadata_store": SqlStoreReference(
                backend="sql_default",
                table_name="openai_files_metadata",
            ).model_dump(exclude_none=True),
        }

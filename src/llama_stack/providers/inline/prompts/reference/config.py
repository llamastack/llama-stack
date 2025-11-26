# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel, Field

from llama_stack.core.datatypes import StackRunConfig


class ReferencePromptsConfig(BaseModel):
    """Configuration for the built-in reference prompt service.

    This provider stores prompts in the configured KVStore (SQLite, PostgreSQL, etc.)
    as specified in the run configuration.
    """

    run_config: StackRunConfig = Field(
        description="Stack run configuration containing storage configuration"
    )

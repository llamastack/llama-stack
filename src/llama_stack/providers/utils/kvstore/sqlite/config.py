# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel, Field

from llama_stack.schema_utils import json_schema_type


@json_schema_type
class SqliteControlPlaneConfig(BaseModel):
    db_path: str = Field(
        description="File path for the sqlite database",
    )
    table_name: str = Field(
        default="llamastack_control_plane",
        description="Table into which all the keys will be placed",
    )

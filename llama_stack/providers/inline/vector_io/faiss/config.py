# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel

from llama_stack.core.storage.datatypes import KVStoreReference
from llama_stack.schema_utils import json_schema_type


@json_schema_type
class FaissVectorIOConfig(BaseModel):
    persistence: KVStoreReference

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "persistence": KVStoreReference(
                backend="kv_default",
                namespace="vector_io::faiss",
            ).model_dump(exclude_none=True)
        }

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Any

from pydantic import BaseModel

from llama_stack.core.storage.datatypes import KVStoreReference


class HuggingfaceDatasetIOConfig(BaseModel):
    kvstore: KVStoreReference

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "kvstore": KVStoreReference(
                backend="kv_default",
                namespace="datasetio::huggingface",
            ).model_dump(exclude_none=True)
        }

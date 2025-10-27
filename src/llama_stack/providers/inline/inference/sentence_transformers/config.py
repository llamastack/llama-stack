# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel


class SentenceTransformersInferenceConfig(BaseModel):
    @classmethod
    def sample_run_config(cls, **kwargs) -> dict[str, Any]:
        return {}

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Literal

from pydantic import BaseModel


class TorchtunePostTrainingConfig(BaseModel):
    torch_seed: int | None = None
    checkpoint_format: Literal["meta", "huggingface"] | None = "meta"

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "checkpoint_format": "meta",
        }

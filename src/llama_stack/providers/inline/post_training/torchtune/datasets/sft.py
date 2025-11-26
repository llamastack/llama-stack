# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Mapping
from typing import Any

import numpy as np
from torch.utils.data import Dataset
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX
from torchtune.data._messages import validate_messages
from torchtune.modules.transforms import Transform

from llama_stack.providers.inline.post_training.torchtune.datasets.format_adapter import (
    llama_stack_chat_to_torchtune_chat,
    llama_stack_instruct_to_torchtune_instruct,
)


class SFTDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        message_transform: Transform,
        model_transform: Transform,
        dataset_type: str,
    ) -> None:
        self._rows = rows
        self._message_transform = message_transform
        self._model_transform = model_transform
        self._dataset_type = dataset_type

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._rows[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        if self._dataset_type == "instruct":
            sample = llama_stack_instruct_to_torchtune_instruct(sample)
        elif self._dataset_type == "dialog":
            sample = llama_stack_chat_to_torchtune_chat(sample)
        else:
            raise ValueError(f"Invalid dataset type: {self._dataset_type}")
        transformed_sample = self._message_transform(sample)
        if "messages" in transformed_sample:
            validate_messages(transformed_sample["messages"])

        tokenized_dict: dict[str, Any] = self._model_transform(transformed_sample)

        if not ("tokens" in tokenized_dict and "mask" in tokenized_dict):
            keys_str = ", ".join(tokenized_dict.keys())
            error_message = (
                f"model_transform returned the following keys: {keys_str}. Must return 'tokens' and 'mask' as keys."
            )
            raise ValueError(error_message)

        # Wherever mask == True, set to CROSS_ENTROPY_IGNORE_IDX. Otherwise keep as tokens
        tokenized_dict["labels"] = list(
            np.where(
                tokenized_dict["mask"],
                CROSS_ENTROPY_IGNORE_IDX,
                tokenized_dict["tokens"],
            )
        )
        assert len(tokenized_dict["tokens"]) == len(tokenized_dict["labels"])

        return tokenized_dict

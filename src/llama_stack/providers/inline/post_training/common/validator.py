# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, IAny, nc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.apis.common.type_system import (
    ChatCompletionInputType,
    DialogType,
    StringType,
)
from llama_stack.providers.utils.common.data_schema_validator import (
    ColumnName,
)

EXPECTED_DATASET_SCHEMA: dict[str, list[dict[str, Any]]] = {
    "instruct": [
        {
            ColumnName.chat_completion_input.value: ChatCompletionInputType(),
            ColumnName.expected_answer.value: StringType(),
        }
    ],
    "dialog": [
        {
            ColumnName.dialog.value: DialogType(),
        }
    ],
}

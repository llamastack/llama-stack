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

import json
from collections.abc import Mapping
from typing import Any

from llama_stack.providers.utils.common.data_schema_validator import ColumnName


def llama_stack_instruct_to_torchtune_instruct(
    sample: Mapping[str, Any],
) -> Mapping[str, Any]:
    assert ColumnName.chat_completion_input.value in sample and ColumnName.expected_answer.value in sample, (
        "Invalid input row"
    )
    input_messages = json.loads(sample[ColumnName.chat_completion_input.value])

    assert len(input_messages) == 1, "llama stack instruct dataset format only supports 1 user message"
    input_message = input_messages[0]

    assert "content" in input_message, "content not found in input message"
    input = input_message["content"]
    output = sample[ColumnName.expected_answer.value]

    return {
        "input": input,
        "output": output,
    }


def llama_stack_chat_to_torchtune_chat(sample: Mapping[str, Any]) -> Mapping[str, Any]:
    assert ColumnName.dialog.value in sample, "Invalid input row"
    role_map = {"user": "human", "assistant": "gpt"}
    dialog = json.loads(sample[ColumnName.dialog.value])

    assert len(dialog) > 1, "dialog must have at least 2 messagse"
    roles = []
    conversations = []
    for message in dialog:
        assert "role" in message and "content" in message, "role and content must in message"
        roles.append(message["role"])
        conversations.append({"from": role_map[message["role"]], "value": message["content"]})

    assert roles[0] == "user", "first message must be from user"
    assert "assistant" in roles, "at least 1 message should be from assistant"

    return {"conversations": conversations}

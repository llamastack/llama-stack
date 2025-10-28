# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Any

from llama_stack.apis.common.type_system import (
    ChatCompletionInputType,
    CompletionInputType,
    StringType,
)
from llama_stack.core.datatypes import Api


class ColumnName(Enum):
    input_query = "input_query"
    expected_answer = "expected_answer"
    chat_completion_input = "chat_completion_input"
    completion_input = "completion_input"
    generated_answer = "generated_answer"
    context = "context"
    dialog = "dialog"
    function = "function"
    language = "language"
    id = "id"
    ground_truth = "ground_truth"


VALID_SCHEMAS_FOR_SCORING = [
    {
        ColumnName.input_query.value: StringType(),
        ColumnName.expected_answer.value: StringType(),
        ColumnName.generated_answer.value: StringType(),
    },
    {
        ColumnName.input_query.value: StringType(),
        ColumnName.expected_answer.value: StringType(),
        ColumnName.generated_answer.value: StringType(),
        ColumnName.context.value: StringType(),
    },
    {
        ColumnName.input_query.value: StringType(),
        ColumnName.expected_answer.value: StringType(),
        ColumnName.generated_answer.value: StringType(),
        ColumnName.function.value: StringType(),
        ColumnName.language.value: StringType(),
        ColumnName.id.value: StringType(),
        ColumnName.ground_truth.value: StringType(),
    },
]

VALID_SCHEMAS_FOR_EVAL = [
    {
        ColumnName.input_query.value: StringType(),
        ColumnName.expected_answer.value: StringType(),
        ColumnName.chat_completion_input.value: ChatCompletionInputType(),
    },
    {
        ColumnName.input_query.value: StringType(),
        ColumnName.expected_answer.value: StringType(),
        ColumnName.completion_input.value: CompletionInputType(),
    },
    {
        ColumnName.input_query.value: StringType(),
        ColumnName.expected_answer.value: StringType(),
        ColumnName.generated_answer.value: StringType(),
        ColumnName.function.value: StringType(),
        ColumnName.language.value: StringType(),
        ColumnName.id.value: StringType(),
        ColumnName.ground_truth.value: StringType(),
    },
]


def get_valid_schemas(api_str: str):
    if api_str == Api.scoring.value:
        return VALID_SCHEMAS_FOR_SCORING
    elif api_str == Api.eval.value:
        return VALID_SCHEMAS_FOR_EVAL
    else:
        raise ValueError(f"Invalid API string: {api_str}")


def validate_dataset_schema(
    dataset_schema: dict[str, Any],
    expected_schemas: list[dict[str, Any]],
):
    if dataset_schema not in expected_schemas:
        raise ValueError(f"Dataset {dataset_schema} does not have a correct input schema in {expected_schemas}")


def validate_row_schema(
    input_row: dict[str, Any],
    expected_schemas: list[dict[str, Any]],
):
    for schema in expected_schemas:
        if all(key in input_row for key in schema):
            return

    raise ValueError(f"Input row {input_row} does not match any of the expected schemas in {expected_schemas}")

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum, StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from llama_stack.apis.resource import Resource, ResourceType
from llama_stack.schema_utils import json_schema_type, register_schema


class DatasetPurpose(StrEnum):
    """
    Purpose of the dataset. Each purpose has a required input data schema.

        {
            "messages": [
                {"role": "user", "content": "Hello, world!"},
                {"role": "assistant", "content": "Hello, world!"},
            ]
        }
        {
            "question": "What is the capital of France?",
            "answer": "Paris"
        }
        {
            "messages": [
                {"role": "user", "content": "Hello, my name is John Doe."},
                {"role": "assistant", "content": "Hello, John Doe. How can I help you today?"},
                {"role": "user", "content": "What's my name?"},
            ],
            "answer": "John Doe"
        }
    """

    post_training_messages = "post-training/messages"
    eval_question_answer = "eval/question-answer"
    eval_messages_answer = "eval/messages-answer"

    # TODO: add more schemas here


class DatasetType(Enum):
    """
    Type of the dataset source.
    """

    uri = "uri"
    rows = "rows"


@json_schema_type
class URIDataSource(BaseModel):
    """A dataset that can be obtained from a URI."""

    type: Literal["uri"] = Field(default="uri", description="The type of data source")
    uri: str = Field(
        ...,
        description="The dataset can be obtained from a URI. E.g. 'https://mywebsite.com/mydata.jsonl', 'lsfs://mydata.jsonl', 'data:csv;base64,{base64_content}'",
    )


@json_schema_type
class RowsDataSource(BaseModel):
    """A dataset stored in rows."""

    type: Literal["rows"] = Field(default="rows", description="The type of data source")
    rows: list[dict[str, Any]] = Field(
        ...,
        description="The dataset is stored in rows. E.g. [{'messages': [{'role': 'user', 'content': 'Hello, world!'}, {'role': 'assistant', 'content': 'Hello, world!'}]}]",
    )


DataSource = Annotated[
    URIDataSource | RowsDataSource,
    Field(discriminator="type"),
]
register_schema(DataSource, name="DataSource")


class CommonDatasetFields(BaseModel):
    """Common fields for a dataset."""

    purpose: DatasetPurpose
    source: DataSource
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional metadata for this dataset",
    )


@json_schema_type
class Dataset(CommonDatasetFields, Resource):
    """Dataset resource for storing and accessing training or evaluation data."""

    type: Literal[ResourceType.dataset] = Field(
        default=ResourceType.dataset, description="Type of resource, always 'dataset' for datasets"
    )

    @property
    def dataset_id(self) -> str:
        return self.identifier

    @property
    def provider_dataset_id(self) -> str | None:
        return self.provider_resource_id


class DatasetInput(CommonDatasetFields, BaseModel):
    """Input parameters for dataset operations."""

    dataset_id: str = Field(..., description="Unique identifier for the dataset")


class ListDatasetsResponse(BaseModel):
    """Response from listing datasets."""

    data: list[Dataset] = Field(..., description="List of datasets")


@json_schema_type
class RegisterDatasetRequest(BaseModel):
    """Request model for registering a dataset."""

    purpose: DatasetPurpose = Field(..., description="The purpose of the dataset")
    source: DataSource = Field(..., description="The data source of the dataset")
    metadata: dict[str, Any] | None = Field(default=None, description="The metadata for the dataset")
    dataset_id: str | None = Field(
        default=None, description="The ID of the dataset. If not provided, an ID will be generated"
    )

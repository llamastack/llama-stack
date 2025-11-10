# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum, StrEnum
from typing import Annotated, Any, Literal, Protocol

from pydantic import BaseModel, Field

from llama_stack.apis.resource import Resource, ResourceType
from llama_stack.apis.version import LLAMA_STACK_API_V1BETA
from llama_stack.schema_utils import json_schema_type, register_schema, webmethod


class DatasetPurpose(StrEnum):
    """
    Purpose of the dataset. Each purpose has a required input data schema.

    :cvar post-training/messages: The dataset contains messages used for post-training.
        {
            "messages": [
                {"role": "user", "content": "Hello, world!"},
                {"role": "assistant", "content": "Hello, world!"},
            ]
        }
    :cvar eval/question-answer: The dataset contains a question column and an answer column.
        {
            "question": "What is the capital of France?",
            "answer": "Paris"
        }
    :cvar eval/messages-answer: The dataset contains a messages column with list of messages and an answer column.
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
    :cvar uri: The dataset can be obtained from a URI.
    :cvar rows: The dataset is stored in rows.
    """

    uri = "uri"
    rows = "rows"


@json_schema_type
class URIDataSource(BaseModel):
    """A dataset that can be obtained from a URI.
    :param uri: The dataset can be obtained from a URI. E.g.
        - "https://mywebsite.com/mydata.jsonl"
        - "lsfs://mydata.jsonl"
        - "data:csv;base64,{base64_content}"
    """

    type: Literal["uri"] = "uri"
    uri: str


@json_schema_type
class RowsDataSource(BaseModel):
    """A dataset stored in rows.
    :param rows: The dataset is stored in rows. E.g.
        - [
            {"messages": [{"role": "user", "content": "Hello, world!"}, {"role": "assistant", "content": "Hello, world!"}]}
        ]
    """

    type: Literal["rows"] = "rows"
    rows: list[dict[str, Any]]


DataSource = Annotated[
    URIDataSource | RowsDataSource,
    Field(discriminator="type"),
]
register_schema(DataSource, name="DataSource")


class CommonDatasetFields(BaseModel):
    """
    Common fields for a dataset.

    :param purpose: Purpose of the dataset indicating its intended use
    :param source: Data source configuration for the dataset
    :param metadata: Additional metadata for the dataset
    """

    purpose: DatasetPurpose
    source: DataSource
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional metadata for this dataset",
    )


@json_schema_type
class Dataset(CommonDatasetFields, Resource):
    """Dataset resource for storing and accessing training or evaluation data.

    :param type: Type of resource, always 'dataset' for datasets
    """

    type: Literal[ResourceType.dataset] = ResourceType.dataset

    @property
    def dataset_id(self) -> str:
        return self.identifier

    @property
    def provider_dataset_id(self) -> str | None:
        return self.provider_resource_id


class DatasetInput(CommonDatasetFields, BaseModel):
    """Input parameters for dataset operations.

    :param dataset_id: Unique identifier for the dataset
    """

    dataset_id: str


class ListDatasetsResponse(BaseModel):
    """Response from listing datasets.

    :param data: List of datasets
    """

    data: list[Dataset]


class Datasets(Protocol):
    @webmethod(route="/datasets", method="POST", level=LLAMA_STACK_API_V1BETA, deprecated=True)
    async def register_dataset(
        self,
        purpose: DatasetPurpose,
        source: DataSource,
        metadata: dict[str, Any] | None = None,
        dataset_id: str | None = None,
    ) -> Dataset:
        """
        Register a new dataset.

        :param purpose: The purpose of the dataset.
        One of:
            - "post-training/messages": The dataset contains a messages column with list of messages for post-training.
                {
                    "messages": [
                        {"role": "user", "content": "Hello, world!"},
                        {"role": "assistant", "content": "Hello, world!"},
                    ]
                }
            - "eval/question-answer": The dataset contains a question column and an answer column for evaluation.
                {
                    "question": "What is the capital of France?",
                    "answer": "Paris"
                }
            - "eval/messages-answer": The dataset contains a messages column with list of messages and an answer column for evaluation.
                {
                    "messages": [
                        {"role": "user", "content": "Hello, my name is John Doe."},
                        {"role": "assistant", "content": "Hello, John Doe. How can I help you today?"},
                        {"role": "user", "content": "What's my name?"},
                    ],
                    "answer": "John Doe"
                }
        :param source: The data source of the dataset. Ensure that the data source schema is compatible with the purpose of the dataset. Examples:
           - {
               "type": "uri",
               "uri": "https://mywebsite.com/mydata.jsonl"
           }
           - {
               "type": "uri",
               "uri": "lsfs://mydata.jsonl"
           }
           - {
               "type": "uri",
               "uri": "data:csv;base64,{base64_content}"
           }
           - {
               "type": "uri",
               "uri": "huggingface://llamastack/simpleqa?split=train"
           }
           - {
               "type": "rows",
               "rows": [
                   {
                       "messages": [
                           {"role": "user", "content": "Hello, world!"},
                           {"role": "assistant", "content": "Hello, world!"},
                       ]
                   }
               ]
           }
        :param metadata: The metadata for the dataset.
           - E.g. {"description": "My dataset"}.
        :param dataset_id: The ID of the dataset. If not provided, an ID will be generated.
        :returns: A Dataset.
        """
        ...

    @webmethod(route="/datasets/{dataset_id:path}", method="GET", level=LLAMA_STACK_API_V1BETA)
    async def get_dataset(
        self,
        dataset_id: str,
    ) -> Dataset:
        """Get a dataset by its ID.

        :param dataset_id: The ID of the dataset to get.
        :returns: A Dataset.
        """
        ...

    @webmethod(route="/datasets", method="GET", level=LLAMA_STACK_API_V1BETA)
    async def list_datasets(self) -> ListDatasetsResponse:
        """List all datasets.

        :returns: A ListDatasetsResponse.
        """
        ...

    @webmethod(route="/datasets/{dataset_id:path}", method="DELETE", level=LLAMA_STACK_API_V1BETA, deprecated=True)
    async def unregister_dataset(
        self,
        dataset_id: str,
    ) -> None:
        """Unregister a dataset by its ID.

        :param dataset_id: The ID of the dataset to unregister.
        """
        ...

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pydantic models for Datasets API requests and responses.

This module defines the request and response models for the Datasets API
using Pydantic with Field descriptions for OpenAPI schema generation.
"""

from enum import Enum, StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from llama_stack_api.resource import Resource, ResourceType
from llama_stack_api.schema_utils import json_schema_type, register_schema


class DatasetPurpose(StrEnum):
    """Purpose of the dataset. Each purpose has a required input data schema."""

    post_training_messages = "post-training/messages"
    """The dataset contains messages used for post-training."""
    eval_question_answer = "eval/question-answer"
    """The dataset contains a question column and an answer column."""
    eval_messages_answer = "eval/messages-answer"
    """The dataset contains a messages column with list of messages and an answer column."""


class DatasetType(Enum):
    """Type of the dataset source."""

    uri = "uri"
    """The dataset can be obtained from a URI."""
    rows = "rows"
    """The dataset is stored in rows."""


@json_schema_type
class URIDataSource(BaseModel):
    """A dataset that can be obtained from a URI."""

    type: Literal["uri"] = Field(default="uri", description="The type of data source.")
    uri: str = Field(
        ...,
        description='The dataset can be obtained from a URI. E.g. "https://mywebsite.com/mydata.jsonl", "lsfs://mydata.jsonl", "data:csv;base64,{base64_content}"',
    )


@json_schema_type
class RowsDataSource(BaseModel):
    """A dataset stored in rows."""

    type: Literal["rows"] = Field(default="rows", description="The type of data source.")
    rows: list[dict[str, Any]] = Field(
        ...,
        description='The dataset is stored in rows. E.g. [{"messages": [{"role": "user", "content": "Hello, world!"}, {"role": "assistant", "content": "Hello, world!"}]}]',
    )


DataSource = Annotated[
    URIDataSource | RowsDataSource,
    Field(discriminator="type"),
]
register_schema(DataSource, name="DataSource")


class CommonDatasetFields(BaseModel):
    """Common fields for a dataset."""

    purpose: DatasetPurpose = Field(..., description="Purpose of the dataset indicating its intended use")
    source: DataSource = Field(..., description="Data source configuration for the dataset")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional metadata for this dataset",
    )


@json_schema_type
class Dataset(CommonDatasetFields, Resource):
    """Dataset resource for storing and accessing training or evaluation data."""

    type: Literal[ResourceType.dataset] = Field(
        default=ResourceType.dataset,
        description="Type of resource, always 'dataset' for datasets",
    )

    @property
    def dataset_id(self) -> str:
        return self.identifier

    @property
    def provider_dataset_id(self) -> str | None:
        return self.provider_resource_id


@json_schema_type
class ListDatasetsResponse(BaseModel):
    """Response from listing datasets."""

    data: list[Dataset] = Field(..., description="List of datasets")


# Request models for each endpoint


@json_schema_type
class RegisterDatasetRequest(BaseModel):
    """Request model for registering a dataset."""

    purpose: DatasetPurpose = Field(..., description="The purpose of the dataset.")
    source: DataSource = Field(..., description="The data source of the dataset.")
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="The metadata for the dataset.",
    )
    dataset_id: str | None = Field(
        default=None,
        description="The ID of the dataset. If not provided, an ID will be generated.",
    )


@json_schema_type
class GetDatasetRequest(BaseModel):
    """Request model for getting a dataset by ID."""

    dataset_id: str = Field(..., description="The ID of the dataset to get.")


@json_schema_type
class UnregisterDatasetRequest(BaseModel):
    """Request model for unregistering a dataset."""

    dataset_id: str = Field(..., description="The ID of the dataset to unregister.")


__all__ = [
    "CommonDatasetFields",
    "Dataset",
    "DatasetPurpose",
    "DatasetType",
    "DataSource",
    "RowsDataSource",
    "URIDataSource",
    "ListDatasetsResponse",
    "RegisterDatasetRequest",
    "GetDatasetRequest",
    "UnregisterDatasetRequest",
]

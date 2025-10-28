# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum, EnumMeta

from pydantic import BaseModel, Field

from llama_stack.schema_utils import json_schema_type


class DynamicApiMeta(EnumMeta):
    def __new__(cls, name, bases, namespace):
        # Store the original enum values
        original_values = {k: v for k, v in namespace.items() if not k.startswith("_")}

        # Create the enum class
        cls = super().__new__(cls, name, bases, namespace)

        # Store the original values for reference
        cls._original_values = original_values
        # Initialize _dynamic_values
        cls._dynamic_values = {}

        return cls

    def __call__(cls, value):
        try:
            return super().__call__(value)
        except ValueError as e:
            # If this value was already dynamically added, return it
            if value in cls._dynamic_values:
                return cls._dynamic_values[value]

            # If the value doesn't exist, create a new enum member
            # Create a new member name from the value
            member_name = value.lower().replace("-", "_")

            # If this member name already exists in the enum, return the existing member
            if member_name in cls._member_map_:
                return cls._member_map_[member_name]

            # Instead of creating a new member, raise ValueError to force users to use Api.add() to
            # register new APIs explicitly
            raise ValueError(f"API '{value}' does not exist. Use Api.add() to register new APIs.") from e

    def __iter__(cls):
        # Allow iteration over both static and dynamic members
        yield from super().__iter__()
        if hasattr(cls, "_dynamic_values"):
            yield from cls._dynamic_values.values()

    def add(cls, value):
        """
        Add a new API to the enum.
        Used to register external APIs.
        """
        member_name = value.lower().replace("-", "_")

        # If this member name already exists in the enum, return it
        if member_name in cls._member_map_:
            return cls._member_map_[member_name]

        # Create a new enum member
        member = object.__new__(cls)
        member._name_ = member_name
        member._value_ = value

        # Add it to the enum class
        cls._member_map_[member_name] = member
        cls._member_names_.append(member_name)
        cls._member_type_ = str

        # Store it in our dynamic values
        cls._dynamic_values[value] = member

        return member


@json_schema_type
class Api(Enum, metaclass=DynamicApiMeta):
    """Enumeration of all available APIs in the Llama Stack system.
    :cvar providers: Provider management and configuration
    :cvar inference: Text generation, chat completions, and embeddings
    :cvar safety: Content moderation and safety shields
    :cvar agents: Agent orchestration and execution
    :cvar batches: Batch processing for asynchronous API requests
    :cvar vector_io: Vector database operations and queries
    :cvar datasetio: Dataset input/output operations
    :cvar scoring: Model output evaluation and scoring
    :cvar eval: Model evaluation and benchmarking framework
    :cvar post_training: Fine-tuning and model training
    :cvar tool_runtime: Tool execution and management
    :cvar telemetry: Observability and system monitoring
    :cvar models: Model metadata and management
    :cvar shields: Safety shield implementations
    :cvar datasets: Dataset creation and management
    :cvar scoring_functions: Scoring function definitions
    :cvar benchmarks: Benchmark suite management
    :cvar tool_groups: Tool group organization
    :cvar files: File storage and management
    :cvar prompts: Prompt versions and management
    :cvar inspect: Built-in system inspection and introspection
    """

    providers = "providers"
    inference = "inference"
    safety = "safety"
    agents = "agents"
    batches = "batches"
    vector_io = "vector_io"
    datasetio = "datasetio"
    scoring = "scoring"
    eval = "eval"
    post_training = "post_training"
    tool_runtime = "tool_runtime"

    models = "models"
    shields = "shields"
    vector_stores = "vector_stores"  # only used for routing table
    datasets = "datasets"
    scoring_functions = "scoring_functions"
    benchmarks = "benchmarks"
    tool_groups = "tool_groups"
    files = "files"
    prompts = "prompts"
    conversations = "conversations"

    # built-in API
    inspect = "inspect"


@json_schema_type
class Error(BaseModel):
    """
    Error response from the API. Roughly follows RFC 7807.

    :param status: HTTP status code
    :param title: Error title, a short summary of the error which is invariant for an error type
    :param detail: Error detail, a longer human-readable description of the error
    :param instance: (Optional) A URL which can be used to retrieve more information about the specific occurrence of the error
    """

    status: int
    title: str
    detail: str
    instance: str | None = None


class ExternalApiSpec(BaseModel):
    """Specification for an external API implementation."""

    module: str = Field(..., description="Python module containing the API implementation")
    name: str = Field(..., description="Name of the API")
    pip_packages: list[str] = Field(default=[], description="List of pip packages to install the API")
    protocol: str = Field(..., description="Name of the protocol class for the API")

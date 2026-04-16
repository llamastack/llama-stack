# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_api import Api, RemoteProviderSpec


def get_provider_spec() -> RemoteProviderSpec:
    return RemoteProviderSpec(
        api=Api.vector_io,
        adapter_type="weaviate",
        provider_type="remote::weaviate",
        pip_packages=[],
        module="llama_stack_provider_vector_io_weaviate",
        config_class="llama_stack_provider_vector_io_weaviate.config.WeaviateVectorIOConfig",
        api_dependencies=[Api.inference],
        optional_api_dependencies=[Api.files, Api.models, Api.file_processors],
        description="""
[Weaviate](https://weaviate.io/) is a vector database provider for Llama Stack.
It allows you to store and query vectors directly within a Weaviate database.
That means you're not limited to storing vectors in memory or in a separate service.

## Features
Weaviate supports:
- Store embeddings and their metadata
- Vector search
- Full-text search
- Hybrid search
- Document storage
- Metadata filtering
- Multi-modal retrieval


## Usage

To use Weaviate in your Llama Stack project, follow these steps:

1. Install the necessary dependencies.
2. Configure your Llama Stack project to use chroma.
3. Start storing and querying vectors.

## Installation

To install Weaviate see the [Weaviate quickstart documentation](https://weaviate.io/developers/weaviate/quickstart).

## Documentation
See [Weaviate's documentation](https://weaviate.io/developers/weaviate) for more details about Weaviate in general.
""",
    )

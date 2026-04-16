# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_api import Api, RemoteProviderSpec


def get_provider_spec() -> RemoteProviderSpec:
    return RemoteProviderSpec(
        api=Api.vector_io,
        adapter_type="chromadb",
        provider_type="remote::chromadb",
        pip_packages=[],
        module="llama_stack_provider_vector_io_chromadb",
        config_class="llama_stack_provider_vector_io_chromadb.config.ChromaVectorIOConfig",
        api_dependencies=[Api.inference],
        optional_api_dependencies=[Api.files, Api.models, Api.file_processors],
        description="""
[Chroma](https://www.trychroma.com/) is an inline and remote vector
database provider for Llama Stack. It allows you to store and query vectors directly within a Chroma database.
That means you're not limited to storing vectors in memory or in a separate service.

## Features
Chroma supports:
- Store embeddings and their metadata
- Vector search
- Full-text search
- Document storage
- Metadata filtering
- Multi-modal retrieval

## Usage

To use Chrome in your Llama Stack project, follow these steps:

1. Install the necessary dependencies.
2. Configure your Llama Stack project to use chroma.
3. Start storing and querying vectors.

## Installation

You can install chroma using pip:

```bash
pip install chromadb
```

## Documentation
See [Chroma's documentation](https://docs.trychroma.com/docs/overview/introduction) for more details about Chroma in general.
""",
    )

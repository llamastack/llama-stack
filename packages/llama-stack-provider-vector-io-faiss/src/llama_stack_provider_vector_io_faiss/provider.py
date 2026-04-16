# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_api import Api, InlineProviderSpec


def get_provider_spec() -> InlineProviderSpec:
    return InlineProviderSpec(
        api=Api.vector_io,
        provider_type="inline::faiss",
        pip_packages=[],
        module="llama_stack_provider_vector_io_faiss",
        config_class="llama_stack_provider_vector_io_faiss.config.FaissVectorIOConfig",
        api_dependencies=[Api.inference],
        optional_api_dependencies=[Api.files, Api.models, Api.file_processors],
        description="""
[Faiss](https://github.com/facebookresearch/faiss) is an inline vector database provider for Llama Stack. It
allows you to store and query vectors directly in memory.
That means you'll get fast and efficient vector retrieval.

## Features

- Lightweight and easy to use
- Fully integrated with Llama Stack
- GPU support
- **Vector search** - FAISS supports pure vector similarity search using embeddings

## Search Modes

**Supported:**
- **Vector Search** (`mode="vector"`): Performs vector similarity search using embeddings

**Not Supported:**
- **Keyword Search** (`mode="keyword"`): Not supported by FAISS
- **Hybrid Search** (`mode="hybrid"`): Not supported by FAISS

> **Note**: FAISS is designed as a pure vector similarity search library. See the [FAISS GitHub repository](https://github.com/facebookresearch/faiss) for more details about FAISS's core functionality.

## Usage

To use Faiss in your Llama Stack project, follow these steps:

1. Install the necessary dependencies.
2. Configure your Llama Stack project to use Faiss.
3. Start storing and querying vectors.

## Installation

You can install Faiss using pip:

```bash
pip install faiss-cpu
```
## Documentation
See [Faiss' documentation](https://faiss.ai/) or the [Faiss Wiki](https://github.com/facebookresearch/faiss/wiki) for
more details about Faiss in general.
""",
    )


def get_deprecated_builtin_spec() -> InlineProviderSpec:
    return InlineProviderSpec(
        api=Api.vector_io,
        provider_type="inline::builtin",
        pip_packages=[],
        module="llama_stack_provider_vector_io_faiss",
        config_class="llama_stack_provider_vector_io_faiss.config.FaissVectorIOConfig",
        deprecation_warning="Please use the `inline::faiss` provider instead.",
        api_dependencies=[Api.inference],
        optional_api_dependencies=[Api.files, Api.models, Api.file_processors],
        description="Deprecated alias for inline::faiss.",
    )

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_api import (
    Api,
    InlineProviderSpec,
    ProviderSpec,
)


def available_providers() -> list[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.file_processors,
            provider_type="inline::pypdf",
            pip_packages=["pypdf"],
            module="llama_stack.providers.inline.file_processor.pypdf",
            config_class="llama_stack.providers.inline.file_processor.pypdf.PyPDFFileProcessorConfig",
            optional_api_dependencies=[Api.files],
            description="""
PyPDF file processor for extracting text content from PDF documents.

## Features

- Simple PDF text extraction using PyPDF library
- Support for password-protected PDFs
- Configurable chunking strategies
- PDF metadata extraction
- Text cleaning and normalization
- Compatible with vector store chunking and embeddings APIs
- Support for both direct file upload and file_id processing (requires Files API)

## Use Cases

- Document processing for RAG applications
- Text extraction from PDF documents
- Content preparation for vector databases
""",
        ),
    ]

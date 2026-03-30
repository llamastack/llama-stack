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
    """Return the list of available file processor provider specifications.

    Returns:
        List of ProviderSpec objects describing available providers
    """
    return [
        InlineProviderSpec(
            api=Api.file_processors,
            provider_type="inline::pypdf",
            pip_packages=["pypdf>=6.7.2"],
            module="llama_stack.providers.inline.file_processor.pypdf",
            config_class="llama_stack.providers.inline.file_processor.pypdf.PyPDFFileProcessorConfig",
            api_dependencies=[Api.files],
            description="PyPDF-based file processor for extracting text content from documents.",
        ),
        InlineProviderSpec(
            api=Api.file_processors,
            provider_type="inline::docling",
            pip_packages=["docling"],
            module="llama_stack.providers.inline.file_processor.docling",
            config_class="llama_stack.providers.inline.file_processor.docling.DoclingFileProcessorConfig",
            api_dependencies=[Api.files],
            description="""
[Docling](https://github.com/docling-project/docling) is a layout-aware, structure-preserving
document parser for Llama Stack. Unlike simple text extraction, Docling understands document
structure — headings, tables, lists, and sections — and produces Markdown-formatted output that
preserves semantic boundaries. It supports PDF, DOCX, PPTX, HTML, and images.

## Features

- **Structure-aware chunking** — splits at semantic boundaries (headings, sections) using Docling's HybridChunker
- **Layout preservation** — tables, lists, and nested structures are converted to Markdown
- **Multi-format support** — PDF, DOCX, PPTX, HTML, and images
- **Better RAG quality** — structured chunks with heading metadata produce more relevant retrieval results

## Usage

Start Llama Stack with the Docling file processor using the `--providers` flag:

```bash
OLLAMA_URL=http://localhost:11434/v1 llama stack run \\
  --providers "file_processors=inline::docling,files=inline::localfs,vector_io=inline::faiss,inference=inline::sentence-transformers,inference=remote::ollama" \\
  --port 8321
```

Or add it to a custom `run.yaml`:

```yaml
file_processors:
  - provider_id: docling
    provider_type: inline::docling
    config: {}
```

## Installation

```bash
pip install docling
```

## Documentation

See [Docling's documentation](https://docling-project.github.io/docling/) for more details.
""",
        ),
    ]

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack_api import FileProcessor, ProcessedContent, VectorStoreChunkingStrategy

from .config import ReferenceFileProcessorImplConfig


class ReferenceFileProcessorImpl(FileProcessor):
    """Reference implementation of the FileProcessor API."""

    def __init__(self, config: ReferenceFileProcessorImplConfig, deps: dict[str, Any]):
        self.config = config
        self.deps = deps

    async def initialize(self) -> None:
        pass

    async def process_file(
        self,
        file_data: bytes,
        filename: str,
        options: dict[str, Any] | None = None,
        chunking_strategy: VectorStoreChunkingStrategy | None = None,
        include_embeddings: bool = False,
    ) -> ProcessedContent:
        """Process a file into structured content."""
        return ProcessedContent(
            content="Placeholder content",
            chunks=None,
            embeddings=None,
            metadata={
                "processor": "reference",
                "filename": filename,
            },
        )

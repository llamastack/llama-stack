# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from fastapi import UploadFile

from llama_stack_api.file_processors import ProcessFileResponse
from llama_stack_api.vector_io import VectorStoreChunkingStrategy

from .config import PyPDFFileProcessorConfig
from .pypdf import PyPDFFileProcessor


class PyPDFFileProcessorAdapter:
    """Adapter for PyPDF file processor."""

    def __init__(self, config: PyPDFFileProcessorConfig, files_api=None) -> None:
        self.config = config
        self.files_api = files_api
        self.processor = PyPDFFileProcessor(config, files_api)

    async def process_file(
        self,
        file: UploadFile | None = None,
        file_id: str | None = None,
        options: dict[str, Any] | None = None,
        chunking_strategy: VectorStoreChunkingStrategy | None = None,
    ) -> ProcessFileResponse:
        """Process a file using PyPDF processor."""
        return await self.processor.process_file(
            file=file,
            file_id=file_id,
            options=options,
            chunking_strategy=chunking_strategy,
        )

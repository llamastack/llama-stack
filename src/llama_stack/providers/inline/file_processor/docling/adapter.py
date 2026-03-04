# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from fastapi import UploadFile

from llama_stack_api.file_processors import ProcessFileRequest, ProcessFileResponse

from .config import DoclingFileProcessorConfig
from .docling import DoclingFileProcessor


class DoclingFileProcessorAdapter:
    """Adapter for Docling file processor."""

    def __init__(self, config: DoclingFileProcessorConfig, files_api=None) -> None:
        self.config = config
        self.files_api = files_api
        self.processor = DoclingFileProcessor(config, files_api)

    async def process_file(
        self,
        request: ProcessFileRequest,
        file: UploadFile | None = None,
    ) -> ProcessFileResponse:
        """Process a file using Docling processor."""
        return await self.processor.process_file(
            file=file,
            file_id=request.file_id,
            options=request.options,
            chunking_strategy=request.chunking_strategy,
        )

    async def shutdown(self) -> None:
        """Shutdown the Docling file processor."""
        pass

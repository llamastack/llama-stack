# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import io
import logging
import time
from typing import Any

from llama_stack.apis.file_processors import FileProcessors, ProcessedContent
from .config import PyPDFConfig

logger = logging.getLogger(__name__)


class PyPDFFileProcessorImpl(FileProcessors):
    def __init__(self, config: PyPDFConfig):
        self.config = config
        logger.info("PyPDF processor initialized")

    async def process_file(
        self, 
        file_data: bytes, 
        filename: str, 
        options: dict[str, Any] | None = None
    ) -> ProcessedContent:
        start_time = time.time()
        logger.info(f"Processing PDF file: {filename}, size: {len(file_data)} bytes")
        
        try:
            # Import here to avoid dependency issues if pypdf not installed
            from pypdf import PdfReader
            
            # Migrate existing 3-line logic from vector_store.py
            pdf_reader = PdfReader(io.BytesIO(file_data))
            text = "\n".join([page.extract_text() for page in pdf_reader.pages])
            
            processing_time = time.time() - start_time
            
            result = ProcessedContent(
                content=text,
                metadata={
                    "pages": len(pdf_reader.pages),
                    "processor": "pypdf",
                    "processing_time_seconds": processing_time,
                    "content_length": len(text),
                    "filename": filename,
                    "file_size_bytes": len(file_data)
                }
            )
            
            logger.info(f"PyPDF processing completed: {len(pdf_reader.pages)} pages, {len(text)} chars, {processing_time:.2f}s")
            return result
            
        except ImportError:
            logger.error("PyPDF not installed. Run: pip install pypdf")
            raise RuntimeError("PyPDF not installed. Run: pip install pypdf")
        except Exception as e:
            logger.error(f"PyPDF processing failed for {filename}: {str(e)}")
            raise RuntimeError(f"PyPDF processing failed: {str(e)}") from e
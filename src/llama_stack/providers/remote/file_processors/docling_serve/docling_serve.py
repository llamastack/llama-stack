# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import aiohttp
import logging
import time
from typing import Any

from llama_stack.apis.file_processors import FileProcessors, ProcessedContent
from .config import DoclingServeConfig

logger = logging.getLogger(__name__)


class DoclingServeFileProcessorImpl(FileProcessors):
    def __init__(self, config: DoclingServeConfig):
        self.config = config
        self.base_url = config.base_url.rstrip('/')
        self.api_key = config.api_key
        self.timeout = config.timeout_seconds
        logger.info(f"DoclingServe processor initialized with endpoint: {self.base_url}")
        
    async def process_file(
        self, 
        file_data: bytes, 
        filename: str, 
        options: dict[str, Any] | None = None
    ) -> ProcessedContent:
        start_time = time.time()
        options = options or {}
        
        logger.info(f"Processing file with DoclingServe: {filename}, size: {len(file_data)} bytes")
        logger.debug(f"DoclingServe options: {options}")
        
        try:
            headers = {'Content-Type': 'application/octet-stream'}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            
            # Prepare request parameters
            params = {
                'filename': filename,
                'output_format': options.get('format', 'markdown'),
            }
            
            # Add other docling-specific options
            if 'extract_tables' in options:
                params['extract_tables'] = str(options['extract_tables']).lower()
            if 'extract_figures' in options:
                params['extract_figures'] = str(options['extract_figures']).lower()
            if 'ocr_enabled' in options:
                params['ocr_enabled'] = str(options['ocr_enabled']).lower()
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(
                    f"{self.base_url}/v1/convert",
                    headers=headers,
                    params=params,
                    data=file_data
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"DoclingServe API error: {response.status} - {error_text}")
                        raise RuntimeError(f"DoclingServe API error: {response.status} - {error_text}")
                    
                    result_data = await response.json()
                    
                    processing_time = time.time() - start_time
                    
                    processed = ProcessedContent(
                        content=result_data.get('content', ''),
                        metadata={
                            'pages': result_data.get('pages', 0),
                            'tables': result_data.get('tables_extracted', 0),
                            'figures': result_data.get('figures_extracted', 0),
                            'format': result_data.get('output_format', 'markdown'),
                            'processor': 'docling_serve',
                            'processing_time_seconds': processing_time,
                            'content_length': len(result_data.get('content', '')),
                            'server_processing_time': result_data.get('server_processing_time'),
                            'server_version': result_data.get('server_version'),
                            'filename': filename,
                            'file_size_bytes': len(file_data)
                        }
                    )
                    
                    logger.info(f"DoclingServe processing completed: {result_data.get('pages', 0)} pages, "
                              f"{result_data.get('tables_extracted', 0)} tables, {processing_time:.2f}s")
                    return processed
                    
        except aiohttp.ClientTimeout:
            logger.error(f"DoclingServe timeout after {self.timeout}s for {filename}")
            raise RuntimeError(f"DoclingServe processing timeout after {self.timeout} seconds")
        except Exception as e:
            logger.error(f"DoclingServe processing failed for {filename}: {str(e)}")
            raise RuntimeError(f"DoclingServe processing failed: {str(e)}") from e
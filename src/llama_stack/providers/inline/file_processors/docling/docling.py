# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import tempfile
import logging
import time
from typing import Any
from pathlib import Path

from llama_stack.apis.file_processors import FileProcessors, ProcessedContent
from .config import DoclingConfig

logger = logging.getLogger(__name__)


class DoclingFileProcessorImpl(FileProcessors):
    def __init__(self, config: DoclingConfig):
        self.config = config
        self.convert_manager = None
        self._initialize_docling()
        logger.info("Docling processor initialized with ConvertManager")
        
    def _initialize_docling(self):
        """Initialize Docling using ConvertManager from docling-jobkit"""
        try:
            from docling_jobkit.convert.manager import ConvertManager
            
            # Initialize ConvertManager with configuration
            manager_config = {
                "cache_dir": self.config.model_cache_dir,
                "enable_gpu": self.config.enable_gpu,
            }
            
            # Remove None values from config
            manager_config = {k: v for k, v in manager_config.items() if v is not None}
            
            self.convert_manager = ConvertManager(**manager_config)
            logger.info("Docling ConvertManager initialized successfully")
            
        except ImportError as e:
            logger.error("Docling JobKit not installed. Run: pip install docling-jobkit")
            raise ImportError("Docling JobKit not installed. Run: pip install docling-jobkit") from e
        except Exception as e:
            logger.error(f"Failed to initialize Docling ConvertManager: {e}")
            raise RuntimeError(f"Failed to initialize Docling ConvertManager: {e}") from e
    
    def _parse_docling_options(self, options: dict[str, Any]) -> dict[str, Any]:
        """Parse and validate Docling-specific options"""
        if not options:
            return {}
        
        # ConvertManager supports these options
        docling_options = {}
        
        # Output format options
        if "format" in options:
            docling_options["output_format"] = options["format"]
        
        # Processing options that ConvertManager handles
        if "extract_tables" in options:
            docling_options["extract_tables"] = bool(options["extract_tables"])
        if "extract_figures" in options:
            docling_options["extract_figures"] = bool(options["extract_figures"])
        if "ocr_enabled" in options:
            docling_options["ocr_enabled"] = bool(options["ocr_enabled"])
        if "ocr_languages" in options and isinstance(options["ocr_languages"], list):
            docling_options["ocr_languages"] = options["ocr_languages"]
        if "preserve_layout" in options:
            docling_options["preserve_layout"] = bool(options["preserve_layout"])
            
        return docling_options
    
    async def process_file(
        self, 
        file_data: bytes, 
        filename: str, 
        options: dict[str, Any] | None = None
    ) -> ProcessedContent:
        start_time = time.time()
        options = options or {}
        
        logger.info(f"Processing file with Docling ConvertManager: {filename}, size: {len(file_data)} bytes")
        logger.debug(f"Docling options: {options}")
        
        try:
            # Parse options for ConvertManager
            docling_options = self._parse_docling_options(options)
            
            # Get converter from ConvertManager
            # This leverages the official docling-jobkit approach
            converter = self.convert_manager.get_converter(**docling_options)
            
            # Process file using temporary file (Docling requirement)
            with tempfile.NamedTemporaryFile(suffix=f"_{filename}", delete=False) as tmp:
                tmp.write(file_data)
                tmp.flush()
                tmp_path = Path(tmp.name)
                
                try:
                    # Convert using the managed converter
                    result = converter.convert(tmp_path)
                    
                    # Determine output format
                    format_type = options.get("format", "markdown")
                    
                    # Export content based on requested format
                    if format_type == "markdown":
                        content = result.document.export_to_markdown()
                    elif format_type == "html": 
                        content = result.document.export_to_html()
                    elif format_type == "json":
                        content = result.document.export_to_json()
                    else:
                        content = result.document.export_to_text()
                    
                    processing_time = time.time() - start_time
                    
                    # Extract metadata from Docling result
                    processed = ProcessedContent(
                        content=content,
                        metadata={
                            "pages": len(result.document.pages) if hasattr(result.document, 'pages') else 0,
                            "tables": len(result.document.tables) if hasattr(result.document, 'tables') else 0,
                            "figures": len(result.document.figures) if hasattr(result.document, 'figures') else 0,
                            "format": format_type,
                            "processor": "docling_jobkit", 
                            "processing_time_seconds": processing_time,
                            "content_length": len(content),
                            "filename": filename,
                            "file_size_bytes": len(file_data),
                            "converter_options": docling_options,
                            "docling_version": getattr(result, 'version', 'unknown'),
                        }
                    )
                    
                    logger.info(f"Docling processing completed: {processed.metadata.get('pages', 0)} pages, "
                              f"{processed.metadata.get('tables', 0)} tables, {processing_time:.2f}s")
                    return processed
                
                finally:
                    # Clean up temporary file
                    try:
                        tmp_path.unlink()
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to cleanup temp file {tmp_path}: {cleanup_error}")
                    
        except Exception as e:
            logger.error(f"Docling processing failed for {filename}: {str(e)}")
            raise RuntimeError(f"Docling processing failed: {str(e)}") from e
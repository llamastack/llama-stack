# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llama_stack.apis.common.content_types import URL, TextContentItem
from llama_stack.apis.tools import RAGDocument
from llama_stack.providers.utils.memory.vector_store import content_from_data_and_mime_type, content_from_doc


async def test_content_from_doc_with_url():
    """Test extracting content from RAGDocument with URL content."""
    mock_url = URL(uri="https://example.com")
    mock_doc = RAGDocument(document_id="foo", content=mock_url)

    mock_response = MagicMock()
    mock_response.text = "Sample content from URL"

    with patch("httpx.AsyncClient") as mock_client:
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance

        result = await content_from_doc(mock_doc)

        assert result == "Sample content from URL"
        mock_instance.get.assert_called_once_with(mock_url.uri)


async def test_content_from_doc_with_pdf_url():
    """Test extracting content from RAGDocument with URL pointing to a PDF."""
    mock_url = URL(uri="https://example.com/document.pdf")
    mock_doc = RAGDocument(document_id="foo", content=mock_url, mime_type="application/pdf")

    mock_response = MagicMock()
    mock_response.content = b"PDF binary data"

    with (
        patch("httpx.AsyncClient") as mock_client,
        patch("llama_stack.providers.utils.memory.vector_store.parse_pdf") as mock_parse_pdf,
    ):
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance
        mock_parse_pdf.return_value = "Extracted PDF content"

        result = await content_from_doc(mock_doc)

        assert result == "Extracted PDF content"
        mock_instance.get.assert_called_once_with(mock_url.uri)
        mock_parse_pdf.assert_called_once_with(b"PDF binary data")


async def test_content_from_doc_with_data_url():
    """Test extracting content from RAGDocument with data URL content."""
    data_url = "data:text/plain;base64,SGVsbG8gV29ybGQ="  # "Hello World" base64 encoded
    mock_url = URL(uri=data_url)
    mock_doc = RAGDocument(document_id="foo", content=mock_url)

    with patch("llama_stack.providers.utils.memory.vector_store.content_from_data") as mock_content_from_data:
        mock_content_from_data.return_value = "Hello World"

        result = await content_from_doc(mock_doc)

        assert result == "Hello World"
        mock_content_from_data.assert_called_once_with(data_url)


async def test_content_from_doc_with_string():
    """Test extracting content from RAGDocument with string content."""
    content_string = "This is plain text content"
    mock_doc = RAGDocument(document_id="foo", content=content_string)

    result = await content_from_doc(mock_doc)

    assert result == content_string


async def test_content_from_doc_with_string_url():
    """Test extracting content from RAGDocument with string URL content."""
    url_string = "https://example.com"
    mock_doc = RAGDocument(document_id="foo", content=url_string)

    mock_response = MagicMock()
    mock_response.text = "Sample content from URL string"

    with patch("httpx.AsyncClient") as mock_client:
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance

        result = await content_from_doc(mock_doc)

        assert result == "Sample content from URL string"
        mock_instance.get.assert_called_once_with(url_string)


async def test_content_from_doc_with_string_pdf_url():
    """Test extracting content from RAGDocument with string URL pointing to a PDF."""
    url_string = "https://example.com/document.pdf"
    mock_doc = RAGDocument(document_id="foo", content=url_string, mime_type="application/pdf")

    mock_response = MagicMock()
    mock_response.content = b"PDF binary data"

    with (
        patch("httpx.AsyncClient") as mock_client,
        patch("llama_stack.providers.utils.memory.vector_store.parse_pdf") as mock_parse_pdf,
    ):
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_instance
        mock_parse_pdf.return_value = "Extracted PDF content from string URL"

        result = await content_from_doc(mock_doc)

        assert result == "Extracted PDF content from string URL"
        mock_instance.get.assert_called_once_with(url_string)
        mock_parse_pdf.assert_called_once_with(b"PDF binary data")


async def test_content_from_doc_with_interleaved_content():
    """Test extracting content from RAGDocument with InterleavedContent (the new case added in the commit)."""
    interleaved_content = [TextContentItem(text="First item"), TextContentItem(text="Second item")]
    mock_doc = RAGDocument(document_id="foo", content=interleaved_content)

    with patch("llama_stack.providers.utils.memory.vector_store.interleaved_content_as_str") as mock_interleaved:
        mock_interleaved.return_value = "First item\nSecond item"

        result = await content_from_doc(mock_doc)

        assert result == "First item\nSecond item"
        mock_interleaved.assert_called_once_with(interleaved_content)


def test_content_from_data_and_mime_type_success_utf8():
    """Test successful decoding with UTF-8 encoding."""
    data = "Hello World! 🌍".encode()
    mime_type = "text/plain"

    with patch("chardet.detect") as mock_detect:
        mock_detect.return_value = {"encoding": "utf-8"}

        result = content_from_data_and_mime_type(data, mime_type)

        mock_detect.assert_called_once_with(data)
        assert result == "Hello World! 🌍"


def test_content_from_data_and_mime_type_error_win1252():
    """Test fallback to UTF-8 when Windows-1252 encoding detection fails."""
    data = "Hello World! 🌍".encode()
    mime_type = "text/plain"

    with patch("chardet.detect") as mock_detect:
        mock_detect.return_value = {"encoding": "Windows-1252"}

        result = content_from_data_and_mime_type(data, mime_type)

        assert result == "Hello World! 🌍"
        mock_detect.assert_called_once_with(data)


def test_content_from_data_and_mime_type_both_encodings_fail():
    """Test that exceptions are raised when both primary and UTF-8 encodings fail."""
    # Create invalid byte sequence that fails with both encodings
    data = b"\xff\xfe\x00\x8f"  # Invalid UTF-8 sequence
    mime_type = "text/plain"

    with patch("chardet.detect") as mock_detect:
        mock_detect.return_value = {"encoding": "windows-1252"}

        # Should raise an exception instead of returning empty string
        with pytest.raises(UnicodeDecodeError):
            content_from_data_and_mime_type(data, mime_type)


async def test_memory_tool_error_handling():
    """Test that memory tool handles various failures gracefully without crashing."""
    from llama_stack.providers.inline.tool_runtime.rag.config import RagToolRuntimeConfig
    from llama_stack.providers.inline.tool_runtime.rag.memory import MemoryToolRuntimeImpl

    config = RagToolRuntimeConfig()
    memory_tool = MemoryToolRuntimeImpl(
        config=config,
        vector_io_api=AsyncMock(),
        inference_api=AsyncMock(),
        files_api=AsyncMock(),
    )

    docs = [
        RAGDocument(document_id="good_doc", content="Good content", metadata={}),
        RAGDocument(document_id="bad_url_doc", content=URL(uri="https://bad.url"), metadata={}),
        RAGDocument(document_id="another_good_doc", content="Another good content", metadata={}),
    ]

    mock_file1 = MagicMock()
    mock_file1.id = "file_good1"
    mock_file2 = MagicMock()
    mock_file2.id = "file_good2"
    memory_tool.files_api.openai_upload_file.side_effect = [mock_file1, mock_file2]

    with patch("httpx.AsyncClient") as mock_client:
        mock_instance = AsyncMock()
        mock_instance.get.side_effect = Exception("Bad URL")
        mock_client.return_value.__aenter__.return_value = mock_instance

        # won't raise exception despite one document failing
        await memory_tool.insert(docs, "vector_store_123")

    # processed 2 documents successfully, skipped 1
    assert memory_tool.files_api.openai_upload_file.call_count == 2
    assert memory_tool.vector_io_api.openai_attach_file_to_vector_store.call_count == 2

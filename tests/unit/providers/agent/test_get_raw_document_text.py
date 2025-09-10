# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import warnings
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llama_stack.apis.agents import Document
from llama_stack.apis.common.content_types import URL, TextContentItem
from llama_stack.providers.inline.agents.meta_reference.agent_instance import get_raw_document_text


async def test_get_raw_document_text_supports_text_mime_types():
    """Test that the function accepts text/* mime types."""
    document = Document(content="Sample text content", mime_type="text/plain")

    result = await get_raw_document_text(document)
    assert result == "Sample text content"


async def test_get_raw_document_text_supports_yaml_mime_type():
    """Test that the function accepts application/yaml mime type."""
    yaml_content = """
    name: test
    version: 1.0
    items:
      - item1
      - item2
    """

    document = Document(content=yaml_content, mime_type="application/yaml")

    result = await get_raw_document_text(document)
    assert result == yaml_content


async def test_get_raw_document_text_supports_deprecated_text_yaml_with_warning():
    """Test that the function accepts text/yaml but emits a deprecation warning."""
    yaml_content = """
    name: test
    version: 1.0
    items:
      - item1
      - item2
    """

    document = Document(content=yaml_content, mime_type="text/yaml")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = await get_raw_document_text(document)

        # Check that result is correct
        assert result == yaml_content

        # Check that exactly one warning was issued
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "text/yaml" in str(w[0].message)
        assert "application/yaml" in str(w[0].message)
        assert "deprecated" in str(w[0].message).lower()


async def test_get_raw_document_text_deprecated_text_yaml_with_url():
    """Test that text/yaml works with URL content and emits warning."""
    yaml_content = "name: test\nversion: 1.0"

    with patch("llama_stack.providers.inline.agents.meta_reference.agent_instance.load_data_from_url") as mock_load:
        mock_load.return_value = yaml_content

        document = Document(content=URL(uri="https://example.com/config.yaml"), mime_type="text/yaml")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = await get_raw_document_text(document)

            # Check that result is correct
            assert result == yaml_content
            mock_load.assert_called_once_with("https://example.com/config.yaml")

            # Check that deprecation warning was issued
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "text/yaml" in str(w[0].message)


async def test_get_raw_document_text_deprecated_text_yaml_with_text_content_item():
    """Test that text/yaml works with TextContentItem and emits warning."""
    yaml_content = "key: value\nlist:\n  - item1\n  - item2"

    document = Document(content=TextContentItem(text=yaml_content), mime_type="text/yaml")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = await get_raw_document_text(document)

        # Check that result is correct
        assert result == yaml_content

        # Check that deprecation warning was issued
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "text/yaml" in str(w[0].message)


async def test_get_raw_document_text_rejects_unsupported_mime_types():
    """Test that the function rejects unsupported mime types."""
    document = Document(
        content="Some content",
        mime_type="application/json",  # Not supported
    )

    with pytest.raises(ValueError, match="Unexpected document mime type: application/json"):
        await get_raw_document_text(document)


async def test_get_raw_document_text_with_url_content():
    """Test that the function handles URL content correctly."""
    mock_response = AsyncMock()
    mock_response.text = "Content from URL"

    with patch("llama_stack.providers.inline.agents.meta_reference.agent_instance.load_data_from_url") as mock_load:
        mock_load.return_value = "Content from URL"

        document = Document(content=URL(uri="https://example.com/test.txt"), mime_type="text/plain")

        result = await get_raw_document_text(document)
        assert result == "Content from URL"
        mock_load.assert_called_once_with("https://example.com/test.txt")


async def test_get_raw_document_text_with_yaml_url():
    """Test that the function handles YAML URLs correctly."""
    yaml_content = "name: test\nversion: 1.0"

    with patch("llama_stack.providers.inline.agents.meta_reference.agent_instance.load_data_from_url") as mock_load:
        mock_load.return_value = yaml_content

        document = Document(content=URL(uri="https://example.com/config.yaml"), mime_type="application/yaml")

        result = await get_raw_document_text(document)
        assert result == yaml_content
        mock_load.assert_called_once_with("https://example.com/config.yaml")


async def test_get_raw_document_text_with_text_content_item():
    """Test that the function handles TextContentItem correctly."""
    document = Document(content=TextContentItem(text="Text content item"), mime_type="text/plain")

    result = await get_raw_document_text(document)
    assert result == "Text content item"


async def test_get_raw_document_text_with_yaml_text_content_item():
    """Test that the function handles YAML TextContentItem correctly."""
    yaml_content = "key: value\nlist:\n  - item1\n  - item2"

    document = Document(content=TextContentItem(text=yaml_content), mime_type="application/yaml")

    result = await get_raw_document_text(document)
    assert result == yaml_content


async def test_get_raw_document_text_rejects_unexpected_content_type():
    """Test that the function rejects unexpected document content types."""
    # Create a mock document that bypasses Pydantic validation
    mock_document = MagicMock(spec=Document)
    mock_document.mime_type = "text/plain"
    mock_document.content = 123  # Unexpected content type (not str, URL, or TextContentItem)

    with pytest.raises(ValueError, match="Unexpected document content type: <class 'int'>"):
        await get_raw_document_text(mock_document)

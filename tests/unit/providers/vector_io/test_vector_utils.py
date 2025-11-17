# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.providers.utils.vector_io.vector_utils import (
    generate_chunk_id,
    sanitize_metadata_for_attributes,
)
from llama_stack_api import Chunk, ChunkMetadata

# This test is a unit test for the chunk_utils.py helpers. This should only contain
# tests which are specific to this file. More general (API-level) tests should be placed in
# tests/integration/vector_io/
#
# How to run this test:
#
# pytest tests/unit/providers/vector_io/test_chunk_utils.py \
# -v -s --tb=short --disable-warnings --asyncio-mode=auto


def test_generate_chunk_id():
    """Test that generate_chunk_id produces expected hashes."""
    chunk_id1 = generate_chunk_id("doc-1", "test")
    chunk_id2 = generate_chunk_id("doc-1", "test ")
    chunk_id3 = generate_chunk_id("doc-1", "test 3")

    chunk_ids = sorted([chunk_id1, chunk_id2, chunk_id3])
    assert chunk_ids == [
        "31d1f9a3-c8d2-66e7-3c37-af2acd329778",
        "d07dade7-29c0-cda7-df29-0249a1dcbc3e",
        "d14f75a1-5855-7f72-2c78-d9fc4275a346",
    ]


def test_generate_chunk_id_with_window():
    """Test that generate_chunk_id with chunk_window produces different IDs."""
    # Create a chunk object to match the original test behavior (passing object to generate_chunk_id)
    chunk = Chunk(content="test", chunk_id="placeholder", metadata={"document_id": "doc-1"})
    chunk_id1 = generate_chunk_id("doc-1", chunk, chunk_window="0-1")
    chunk_id2 = generate_chunk_id("doc-1", chunk, chunk_window="1-2")
    # Verify that different windows produce different IDs
    assert chunk_id1 != chunk_id2
    assert len(chunk_id1) == 36  # Valid UUID format
    assert len(chunk_id2) == 36  # Valid UUID format


def test_chunk_creation_with_explicit_id():
    """Test that chunks can be created with explicit chunk_id."""
    chunk_id = generate_chunk_id("doc-1", "test")
    chunk = Chunk(
        content="test",
        chunk_id=chunk_id,
        metadata={"document_id": "doc-1"},
    )
    assert chunk.chunk_id == chunk_id
    assert chunk.chunk_id == "31d1f9a3-c8d2-66e7-3c37-af2acd329778"


def test_chunk_with_metadata():
    """Test chunks with ChunkMetadata."""
    chunk_id = "chunk-id-1"
    chunk = Chunk(
        content="test",
        chunk_id=chunk_id,
        metadata={"document_id": "existing-id"},
        chunk_metadata=ChunkMetadata(document_id="document_1"),
    )
    assert chunk.chunk_id == "chunk-id-1"
    assert chunk.document_id == "existing-id"  # metadata takes precedence


def test_chunk_serialization():
    """Test that chunk_id is properly serialized."""
    chunk = Chunk(
        content="test",
        chunk_id="test-chunk-id",
        metadata={"document_id": "doc-1"},
    )
    serialized_chunk = chunk.model_dump()
    assert serialized_chunk["chunk_id"] == "test-chunk-id"
    assert "chunk_id" in serialized_chunk


def test_sanitize_metadata_for_attributes():
    """Test sanitization of metadata for VectorStoreSearchResponse.attributes."""
    # metadata with lists should be converted to strings
    metadata = {
        "tags": ["transformers", "h100-compatible", "region:us"],
        "model_name": "granite-3.3-8b",
        "score": 0.95,
        "active": True,
        "count": 42,
        "nested": {"key": "value"},  # Should be filtered out
    }
    result = sanitize_metadata_for_attributes(metadata)

    # Lists converted to comma-separated strings
    assert result["tags"] == "transformers, h100-compatible, region:us"
    # Primitives preserved
    assert result["model_name"] == "granite-3.3-8b"
    assert result["score"] == 0.95
    assert result["active"] is True
    assert result["count"] == 42.0  # int -> float
    # Complex types filtered out
    assert "nested" not in result

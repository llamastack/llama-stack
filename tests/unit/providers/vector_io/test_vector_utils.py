# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import time

from llama_stack.providers.utils.vector_io.vector_utils import (
    generate_chunk_id,
    load_embedded_chunk_with_backward_compat,
)
from llama_stack_api import Chunk, ChunkMetadata, EmbeddedChunk, VectorStoreFileObject

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
    chunk = Chunk(
        content="test",
        chunk_id="placeholder",
        metadata={"document_id": "doc-1"},
        chunk_metadata=ChunkMetadata(
            document_id="doc-1",
            chunk_id="placeholder",
            created_timestamp=int(time.time()),
            updated_timestamp=int(time.time()),
            content_token_count=1,
        ),
    )
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
        chunk_metadata=ChunkMetadata(
            document_id="doc-1",
            chunk_id=chunk_id,
            created_timestamp=int(time.time()),
            updated_timestamp=int(time.time()),
            content_token_count=1,
        ),
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
        chunk_metadata=ChunkMetadata(
            document_id="document_1",
            chunk_id=chunk_id,
            created_timestamp=int(time.time()),
            updated_timestamp=int(time.time()),
            content_token_count=1,
        ),
    )
    assert chunk.chunk_id == "chunk-id-1"
    assert chunk.document_id == "existing-id"  # metadata takes precedence


def test_chunk_serialization():
    """Test that chunk_id is properly serialized."""
    chunk = Chunk(
        content="test",
        chunk_id="test-chunk-id",
        metadata={"document_id": "doc-1"},
        chunk_metadata=ChunkMetadata(
            document_id="doc-1",
            chunk_id="test-chunk-id",
            created_timestamp=int(time.time()),
            updated_timestamp=int(time.time()),
            content_token_count=1,
        ),
    )
    serialized_chunk = chunk.model_dump()
    assert serialized_chunk["chunk_id"] == "test-chunk-id"
    assert "chunk_id" in serialized_chunk


def test_vector_store_file_object_attributes_validation():
    """Test VectorStoreFileObject validates and sanitizes attributes at input boundary."""
    # Test with metadata containing lists, nested dicts, and primitives
    from llama_stack_api.vector_io import VectorStoreChunkingStrategyAuto

    file_obj = VectorStoreFileObject(
        id="file-123",
        attributes={
            "tags": ["transformers", "h100-compatible", "region:us"],  # List -> string
            "model_name": "granite-3.3-8b",  # String preserved
            "score": 0.95,  # Float preserved
            "active": True,  # Bool preserved
            "count": 42,  # Int -> float
            "nested": {"key": "value"},  # Dict filtered out
        },
        chunking_strategy=VectorStoreChunkingStrategyAuto(),
        created_at=1234567890,
        status="completed",
        vector_store_id="vs-123",
    )

    # Lists converted to comma-separated strings
    assert file_obj.attributes["tags"] == "transformers, h100-compatible, region:us"
    # Primitives preserved
    assert file_obj.attributes["model_name"] == "granite-3.3-8b"
    assert file_obj.attributes["score"] == 0.95
    assert file_obj.attributes["active"] is True
    assert file_obj.attributes["count"] == 42.0  # int -> float
    # Complex types filtered out
    assert "nested" not in file_obj.attributes


def test_vector_store_file_object_attributes_constraints():
    """Test VectorStoreFileObject enforces OpenAPI constraints on attributes."""
    from llama_stack_api.vector_io import VectorStoreChunkingStrategyAuto

    # Test max 16 properties
    many_attrs = {f"key{i}": f"value{i}" for i in range(20)}
    file_obj = VectorStoreFileObject(
        id="file-123",
        attributes=many_attrs,
        chunking_strategy=VectorStoreChunkingStrategyAuto(),
        created_at=1234567890,
        status="completed",
        vector_store_id="vs-123",
    )
    assert len(file_obj.attributes) == 16  # Max 16 properties

    # Test max 64 char keys are filtered
    long_key_attrs = {"a" * 65: "value", "valid_key": "value"}
    file_obj = VectorStoreFileObject(
        id="file-124",
        attributes=long_key_attrs,
        chunking_strategy=VectorStoreChunkingStrategyAuto(),
        created_at=1234567890,
        status="completed",
        vector_store_id="vs-123",
    )
    assert "a" * 65 not in file_obj.attributes
    assert "valid_key" in file_obj.attributes

    # Test max 512 char string values are truncated
    long_value_attrs = {"key": "x" * 600}
    file_obj = VectorStoreFileObject(
        id="file-125",
        attributes=long_value_attrs,
        chunking_strategy=VectorStoreChunkingStrategyAuto(),
        created_at=1234567890,
        status="completed",
        vector_store_id="vs-123",
    )
    assert len(file_obj.attributes["key"]) == 512


# ===== BACKWARD COMPATIBILITY TESTS =====


def test_load_embedded_chunk_current_format():
    """Test loading chunk data in current format (top-level embedding fields)"""
    timestamp = int(time.time())
    current_data = {
        "chunk_id": "test_chunk_current",
        "content": "This is current format content",
        "metadata": {"source": "current_system"},
        "chunk_metadata": {
            "document_id": "doc_current",
            "chunk_id": "test_chunk_current",
            "created_timestamp": timestamp,
            "updated_timestamp": timestamp,
            "content_token_count": 5,
        },
        "embedding": [0.1, 0.2, 0.3],
        "embedding_model": "text-embedding-3-small",
        "embedding_dimension": 3,
    }

    chunk = load_embedded_chunk_with_backward_compat(current_data)

    assert isinstance(chunk, EmbeddedChunk)
    assert chunk.embedding_model == "text-embedding-3-small"
    assert chunk.embedding_dimension == 3
    assert chunk.chunk_id == "test_chunk_current"
    assert chunk.content == "This is current format content"


def test_load_embedded_chunk_legacy_format():
    """Test loading chunk data in legacy format (embedding fields in chunk_metadata)"""
    timestamp = int(time.time())
    legacy_data = {
        "chunk_id": "test_chunk_legacy",
        "content": "This is legacy format content",
        "metadata": {"source": "legacy_system"},
        "chunk_metadata": {
            "document_id": "doc_legacy",
            "chunk_id": "test_chunk_legacy",
            "created_timestamp": timestamp,
            "updated_timestamp": timestamp,
            "content_token_count": 5,
            # Legacy field locations
            "chunk_embedding_model": "text-embedding-ada-002",
            "chunk_embedding_dimension": 3,
        },
        "embedding": [0.4, 0.5, 0.6],
        # Note: no top-level embedding_model or embedding_dimension
    }

    chunk = load_embedded_chunk_with_backward_compat(legacy_data)

    assert isinstance(chunk, EmbeddedChunk)
    assert chunk.embedding_model == "text-embedding-ada-002"  # Migrated from chunk_metadata
    assert chunk.embedding_dimension == 3  # Migrated from chunk_metadata
    assert chunk.chunk_id == "test_chunk_legacy"
    assert chunk.content == "This is legacy format content"


def test_load_embedded_chunk_missing_model():
    """Test fallback to 'unknown' when embedding_model is missing"""
    timestamp = int(time.time())
    missing_model_data = {
        "chunk_id": "test_chunk_missing_model",
        "content": "Content with missing model",
        "metadata": {"source": "incomplete_system"},
        "chunk_metadata": {
            "document_id": "doc_incomplete",
            "chunk_id": "test_chunk_missing_model",
            "created_timestamp": timestamp,
            "updated_timestamp": timestamp,
            "content_token_count": 4,
            # No embedding model info anywhere
        },
        "embedding": [0.7, 0.8, 0.9],
        "embedding_dimension": 3,  # Dimension present at top level
    }

    chunk = load_embedded_chunk_with_backward_compat(missing_model_data)

    assert isinstance(chunk, EmbeddedChunk)
    assert chunk.embedding_model == "unknown"  # Fallback value
    assert chunk.embedding_dimension == 3  # Preserved from top level
    assert chunk.chunk_id == "test_chunk_missing_model"


def test_load_embedded_chunk_infer_dimension():
    """Test dimension inference from embedding vector length"""
    timestamp = int(time.time())
    missing_dimension_data = {
        "chunk_id": "test_chunk_infer_dim",
        "content": "Content with inferred dimension",
        "metadata": {"source": "partial_system"},
        "chunk_metadata": {
            "document_id": "doc_partial",
            "chunk_id": "test_chunk_infer_dim",
            "created_timestamp": timestamp,
            "updated_timestamp": timestamp,
            "content_token_count": 4,
            "chunk_embedding_model": "some-embedding-model",  # Model present in chunk_metadata
            # No dimension info anywhere
        },
        "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],  # 5-dimensional vector
    }

    chunk = load_embedded_chunk_with_backward_compat(missing_dimension_data)

    assert isinstance(chunk, EmbeddedChunk)
    assert chunk.embedding_model == "some-embedding-model"  # Migrated from chunk_metadata
    assert chunk.embedding_dimension == 5  # Inferred from embedding length
    assert chunk.chunk_id == "test_chunk_infer_dim"


def test_load_embedded_chunk_missing_embedding_vector():
    """Test handling when embedding vector is missing entirely"""
    timestamp = int(time.time())
    no_embedding_data = {
        "chunk_id": "test_chunk_no_embedding",
        "content": "Content with no embedding",
        "metadata": {"source": "broken_system"},
        "chunk_metadata": {
            "document_id": "doc_broken",
            "chunk_id": "test_chunk_no_embedding",
            "created_timestamp": timestamp,
            "updated_timestamp": timestamp,
            "content_token_count": 4,
        },
        # No embedding vector, no model, no dimension
    }

    chunk = load_embedded_chunk_with_backward_compat(no_embedding_data)

    assert isinstance(chunk, EmbeddedChunk)
    assert chunk.embedding_model == "unknown"  # Fallback value
    assert chunk.embedding_dimension == 0  # Fallback value
    assert chunk.chunk_id == "test_chunk_no_embedding"


def test_load_embedded_chunk_clean_interface():
    """Test that function has clean interface without unused parameters"""
    timestamp = int(time.time())
    legacy_data = {
        "chunk_id": "test_chunk_clean",
        "content": "Test clean interface",
        "metadata": {"source": "test"},
        "chunk_metadata": {
            "document_id": "doc_test",
            "chunk_id": "test_chunk_clean",
            "created_timestamp": timestamp,
            "updated_timestamp": timestamp,
            "content_token_count": 3,
            "chunk_embedding_model": "test-model",
            # No dimension - will be inferred
        },
        "embedding": [0.1, 0.2],
    }

    chunk = load_embedded_chunk_with_backward_compat(legacy_data)

    # Verify the chunk was created correctly
    assert chunk.embedding_model == "test-model"
    assert chunk.embedding_dimension == 2


def test_load_embedded_chunk_without_logging():
    """Test that function works without logging when no logger provided"""
    timestamp = int(time.time())
    simple_data = {
        "chunk_id": "test_chunk_no_logging",
        "content": "Test no logging",
        "metadata": {"source": "test"},
        "chunk_metadata": {
            "document_id": "doc_test",
            "chunk_id": "test_chunk_no_logging",
            "created_timestamp": timestamp,
            "updated_timestamp": timestamp,
            "content_token_count": 3,
        },
        "embedding": [0.1],
        "embedding_model": "current-model",
        "embedding_dimension": 1,
    }

    # Should work without issues even with no logger
    chunk = load_embedded_chunk_with_backward_compat(simple_data)

    assert chunk.embedding_model == "current-model"
    assert chunk.embedding_dimension == 1


def test_load_embedded_chunk_modifies_input():
    """Test that the function modifies input dict directly (FAISS pattern)"""
    timestamp = int(time.time())
    data = {
        "chunk_id": "test_chunk_modify",
        "content": "Test modification",
        "metadata": {"source": "test"},
        "chunk_metadata": {
            "document_id": "doc_test",
            "chunk_id": "test_chunk_modify",
            "created_timestamp": timestamp,
            "updated_timestamp": timestamp,
            "content_token_count": 2,
            "chunk_embedding_model": "modified-model",
        },
        "embedding": [0.1, 0.2],
    }

    chunk = load_embedded_chunk_with_backward_compat(data)

    # Function should have modified the input dict (efficient approach)
    assert chunk.embedding_model == "modified-model"
    assert chunk.embedding_dimension == 2
    assert data["embedding_model"] == "modified-model"  # Input was modified
    assert data["embedding_dimension"] == 2  # Input was modified


def test_load_embedded_chunk_complex_migration():
    """Test complex migration scenario with mixed field locations"""
    timestamp = int(time.time())
    mixed_data = {
        "chunk_id": "test_chunk_mixed",
        "content": "Mixed field locations",
        "metadata": {"source": "mixed_system", "extra": "data"},
        "chunk_metadata": {
            "document_id": "doc_mixed",
            "chunk_id": "test_chunk_mixed",
            "created_timestamp": timestamp,
            "updated_timestamp": timestamp,
            "content_token_count": 3,
            "chunk_embedding_model": "legacy-model",  # In old location
            "extra_metadata": "preserved",
        },
        "embedding": [0.1, 0.2, 0.3, 0.4],
        "embedding_dimension": 4,  # In new location (should take precedence)
        # No top-level embedding_model (should be migrated)
        "extra_field": "also_preserved",
    }

    chunk = load_embedded_chunk_with_backward_compat(mixed_data)

    assert isinstance(chunk, EmbeddedChunk)
    assert chunk.embedding_model == "legacy-model"  # Migrated from chunk_metadata
    assert chunk.embedding_dimension == 4  # Used from top-level (no migration needed)
    assert chunk.chunk_id == "test_chunk_mixed"
    assert chunk.content == "Mixed field locations"
    # Verify other fields are preserved
    assert "extra" in chunk.metadata
    assert chunk.metadata["extra"] == "data"

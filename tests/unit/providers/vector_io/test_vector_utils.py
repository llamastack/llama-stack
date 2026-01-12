# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

<<<<<<< HEAD
from llama_stack.apis.vector_io import Chunk, ChunkMetadata
from llama_stack.providers.utils.vector_io.vector_utils import generate_chunk_id
=======
import time

from llama_stack.providers.utils.vector_io.vector_utils import (
    generate_chunk_id,
    load_embedded_chunk_with_backward_compat,
)
from llama_stack_api import Chunk, ChunkMetadata, VectorStoreFileObject
>>>>>>> 7d821e02 (chore: Add backwards compatibility for Milvus Chunks (#4484))

# This test is a unit test for the chunk_utils.py helpers. This should only contain
# tests which are specific to this file. More general (API-level) tests should be placed in
# tests/integration/vector_io/
#
# How to run this test:
#
# pytest tests/unit/providers/vector_io/test_chunk_utils.py \
# -v -s --tb=short --disable-warnings --asyncio-mode=auto


def test_generate_chunk_id():
    chunks = [
        Chunk(content="test", metadata={"document_id": "doc-1"}),
        Chunk(content="test ", metadata={"document_id": "doc-1"}),
        Chunk(content="test 3", metadata={"document_id": "doc-1"}),
    ]

    chunk_ids = sorted([chunk.chunk_id for chunk in chunks])
    assert chunk_ids == [
        "31d1f9a3-c8d2-66e7-3c37-af2acd329778",
        "d07dade7-29c0-cda7-df29-0249a1dcbc3e",
        "d14f75a1-5855-7f72-2c78-d9fc4275a346",
    ]


def test_generate_chunk_id_with_window():
    chunk = Chunk(content="test", metadata={"document_id": "doc-1"})
    chunk_id1 = generate_chunk_id("doc-1", chunk, chunk_window="0-1")
    chunk_id2 = generate_chunk_id("doc-1", chunk, chunk_window="1-2")
    assert chunk_id1 == "8630321a-d9cb-2bb6-cd28-ebf68dafd866"
    assert chunk_id2 == "13a1c09a-cbda-b61a-2d1a-7baa90888685"


def test_chunk_id():
    # Test with existing chunk ID
    chunk_with_id = Chunk(content="test", metadata={"document_id": "existing-id"})
    assert chunk_with_id.chunk_id == "11704f92-42b6-61df-bf85-6473e7708fbd"

    # Test with document ID in metadata
    chunk_with_doc_id = Chunk(content="test", metadata={"document_id": "doc-1"})
    assert chunk_with_doc_id.chunk_id == generate_chunk_id("doc-1", "test")

    # Test chunks with ChunkMetadata
    chunk_with_metadata = Chunk(
        content="test",
        metadata={"document_id": "existing-id", "chunk_id": "chunk-id-1"},
        chunk_metadata=ChunkMetadata(document_id="document_1"),
    )
    assert chunk_with_metadata.chunk_id == "chunk-id-1"

    # Test with no ID or document ID
    chunk_without_id = Chunk(content="test")
    generated_id = chunk_without_id.chunk_id
    assert isinstance(generated_id, str) and len(generated_id) == 36  # Should be a valid UUID


<<<<<<< HEAD
def test_stored_chunk_id_alias():
    # Test with existing chunk ID alias
    chunk_with_alias = Chunk(content="test", metadata={"document_id": "existing-id", "chunk_id": "chunk-id-1"})
    assert chunk_with_alias.chunk_id == "chunk-id-1"
    serialized_chunk = chunk_with_alias.model_dump()
    assert serialized_chunk["stored_chunk_id"] == "chunk-id-1"
    # showing chunk_id is not serialized (i.e., a computed field)
    assert "chunk_id" not in serialized_chunk
    assert chunk_with_alias.stored_chunk_id == "chunk-id-1"
=======
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


def test_load_embedded_chunk_backward_compatibility():
    """Test backward compatibility migration from legacy to current format"""
    timestamp = int(time.time())

    # Test current format (no migration needed)
    current_data = {
        "chunk_id": "current",
        "content": "test",
        "metadata": {},
        "chunk_metadata": {
            "document_id": "doc1",
            "chunk_id": "current",
            "created_timestamp": timestamp,
            "updated_timestamp": timestamp,
            "content_token_count": 1,
        },
        "embedding": [0.1, 0.2, 0.3],
        "embedding_model": "current-model",
        "embedding_dimension": 3,
    }
    chunk = load_embedded_chunk_with_backward_compat(current_data)
    assert chunk.embedding_model == "current-model"
    assert chunk.embedding_dimension == 3

    # Test legacy format (fields in chunk_metadata)
    legacy_data = {
        "chunk_id": "legacy",
        "content": "test",
        "metadata": {},
        "chunk_metadata": {
            "document_id": "doc1",
            "chunk_id": "legacy",
            "created_timestamp": timestamp,
            "updated_timestamp": timestamp,
            "content_token_count": 1,
            "chunk_embedding_model": "legacy-model",
            "chunk_embedding_dimension": 3,
        },
        "embedding": [0.4, 0.5, 0.6],
    }
    chunk = load_embedded_chunk_with_backward_compat(legacy_data)
    assert chunk.embedding_model == "legacy-model"  # Migrated
    assert chunk.embedding_dimension == 3  # Migrated


def test_load_embedded_chunk_fallbacks():
    """Test fallback behavior when embedding metadata is missing"""
    timestamp = int(time.time())

    # Test missing model (should fallback to "unknown")
    base_data = {
        "chunk_id": "fallback",
        "content": "test",
        "metadata": {},
        "chunk_metadata": {
            "document_id": "doc1",
            "chunk_id": "fallback",
            "created_timestamp": timestamp,
            "updated_timestamp": timestamp,
            "content_token_count": 1,
        },
        "embedding": [0.1, 0.2],
    }
    chunk = load_embedded_chunk_with_backward_compat(base_data)
    assert chunk.embedding_model == "unknown"
    assert chunk.embedding_dimension == 2  # Inferred from embedding length

    # Test missing embedding vector (should add empty list)
    no_embedding_data = {
        "chunk_id": "fallback",
        "content": "test",
        "metadata": {},
        "chunk_metadata": {
            "document_id": "doc1",
            "chunk_id": "fallback",
            "created_timestamp": timestamp,
            "updated_timestamp": timestamp,
            "content_token_count": 1,
        },
    }
    chunk = load_embedded_chunk_with_backward_compat(no_embedding_data)
    assert chunk.embedding_model == "unknown"
    assert chunk.embedding_dimension == 0
    assert chunk.embedding == []
>>>>>>> 7d821e02 (chore: Add backwards compatibility for Milvus Chunks (#4484))

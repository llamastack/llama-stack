# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.vector_io import Chunk, ChunkMetadata
from llama_stack.providers.utils.vector_io.vector_utils import generate_chunk_id

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


def test_stored_chunk_id_alias():
    # Test with existing chunk ID alias
    chunk_with_alias = Chunk(content="test", metadata={"document_id": "existing-id", "chunk_id": "chunk-id-1"})
    assert chunk_with_alias.chunk_id == "chunk-id-1"
    serialized_chunk = chunk_with_alias.model_dump()
    assert serialized_chunk["stored_chunk_id"] == "chunk-id-1"
    # showing chunk_id is not serialized (i.e., a computed field)
    assert "chunk_id" not in serialized_chunk
    assert chunk_with_alias.stored_chunk_id == "chunk-id-1"

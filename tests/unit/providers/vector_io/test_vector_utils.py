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
        "177a1368-f6a8-0c50-6e92-18677f2c3de3",
        "bc744db3-1b25-0a9c-cdff-b6ba3df73c36",
        "f68df25d-d9aa-ab4d-5684-64a233add20d",
    ]


def test_generate_chunk_id_with_window():
    chunk = Chunk(content="test", metadata={"document_id": "doc-1"})
    chunk_id1 = generate_chunk_id("doc-1", chunk, chunk_window="0-1")
    chunk_id2 = generate_chunk_id("doc-1", chunk, chunk_window="1-2")
    assert chunk_id1 == "149018fe-d0eb-0f8d-5f7f-726bdd2aeedb"
    assert chunk_id2 == "4562c1ee-9971-1f3b-51a6-7d05e5211154"


def test_chunk_id():
    # Test with existing chunk ID
    chunk_with_id = Chunk(content="test", metadata={"document_id": "existing-id"})
    assert chunk_with_id.chunk_id == "84ededcc-b80b-a83e-1a20-ca6515a11350"

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

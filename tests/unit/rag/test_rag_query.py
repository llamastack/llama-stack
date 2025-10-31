# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock

import pytest

from llama_stack.apis.tools.rag_tool import RAGQueryConfig
from llama_stack.apis.vector_io import (
    Chunk,
    ChunkMetadata,
    QueryChunksResponse,
)
from llama_stack.providers.inline.tool_runtime.rag.memory import MemoryToolRuntimeImpl


class TestRagQuery:
    async def test_query_raises_on_empty_vector_store_ids(self):
        rag_tool = MemoryToolRuntimeImpl(
            config=MagicMock(), vector_io_api=MagicMock(), inference_api=MagicMock(), files_api=MagicMock()
        )
        with pytest.raises(ValueError):
            await rag_tool.query(content=MagicMock(), vector_store_ids=[])

    async def test_query_chunk_metadata_handling(self):
        rag_tool = MemoryToolRuntimeImpl(
            config=MagicMock(), vector_io_api=MagicMock(), inference_api=MagicMock(), files_api=MagicMock()
        )
        content = "test query content"
        vector_store_ids = ["db1"]

        chunk_metadata = ChunkMetadata(
            document_id="doc1",
            chunk_id="chunk1",
            source="test_source",
            metadata_token_count=5,
        )
        interleaved_content = MagicMock()
        chunk = Chunk(
            content=interleaved_content,
            chunk_id="chunk1",
            metadata={
                "key1": "value1",
                "token_count": 10,
                "metadata_token_count": 5,
                # Note this is inserted into `metadata` during MemoryToolRuntimeImpl().insert()
                "document_id": "doc1",
            },
            chunk_metadata=chunk_metadata,
        )

        query_response = QueryChunksResponse(chunks=[chunk], scores=[1.0])

        rag_tool.vector_io_api.query_chunks = AsyncMock(return_value=query_response)
        result = await rag_tool.query(content=content, vector_store_ids=vector_store_ids)

        assert result is not None
        expected_metadata_string = (
            "Metadata: {'chunk_id': 'chunk1', 'document_id': 'doc1', 'source': 'test_source', 'key1': 'value1'}"
        )
        assert expected_metadata_string in result.content[1].text
        assert result.content is not None

    async def test_query_raises_incorrect_mode(self):
        with pytest.raises(ValueError):
            RAGQueryConfig(mode="invalid_mode")

    async def test_query_accepts_valid_modes(self):
        default_config = RAGQueryConfig()  # Test default (vector)
        assert default_config.mode == "vector"
        vector_config = RAGQueryConfig(mode="vector")  # Test vector
        assert vector_config.mode == "vector"
        keyword_config = RAGQueryConfig(mode="keyword")  # Test keyword
        assert keyword_config.mode == "keyword"
        hybrid_config = RAGQueryConfig(mode="hybrid")  # Test hybrid
        assert hybrid_config.mode == "hybrid"

        # Test that invalid mode raises an error
        with pytest.raises(ValueError):
            RAGQueryConfig(mode="wrong_mode")

    async def test_query_adds_vector_store_id_to_chunk_metadata(self):
        rag_tool = MemoryToolRuntimeImpl(
            config=MagicMock(),
            vector_io_api=MagicMock(),
            inference_api=MagicMock(),
            files_api=MagicMock(),
        )

        vector_store_ids = ["db1", "db2"]

        # Fake chunks from each DB
        chunk_metadata1 = ChunkMetadata(
            document_id="doc1",
            chunk_id="chunk1",
            source="test_source1",
            metadata_token_count=5,
        )
        chunk1 = Chunk(
            content="chunk from db1",
            chunk_id="c1",
            metadata={"vector_store_id": "db1", "document_id": "doc1"},
            chunk_metadata=chunk_metadata1,
        )

        chunk_metadata2 = ChunkMetadata(
            document_id="doc2",
            chunk_id="chunk2",
            source="test_source2",
            metadata_token_count=5,
        )
        chunk2 = Chunk(
            content="chunk from db2",
            chunk_id="c2",
            metadata={"vector_store_id": "db2", "document_id": "doc2"},
            chunk_metadata=chunk_metadata2,
        )

        rag_tool.vector_io_api.query_chunks = AsyncMock(
            side_effect=[
                QueryChunksResponse(chunks=[chunk1], scores=[0.9]),
                QueryChunksResponse(chunks=[chunk2], scores=[0.8]),
            ]
        )

        result = await rag_tool.query(content="test", vector_store_ids=vector_store_ids)
        returned_chunks = result.metadata["chunks"]
        returned_scores = result.metadata["scores"]
        returned_doc_ids = result.metadata["document_ids"]
        returned_vector_store_ids = result.metadata["vector_store_ids"]

        assert returned_chunks == ["chunk from db1", "chunk from db2"]
        assert returned_scores == (0.9, 0.8)
        assert returned_doc_ids == ["doc1", "doc2"]
        assert returned_vector_store_ids == ["db1", "db2"]

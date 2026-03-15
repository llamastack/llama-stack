# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from llama_stack_api import (
    EmbeddedChunk,
    InsertChunksRequest,
    QueryChunksRequest,
    VectorStore,
)
from llama_stack.providers.remote.vector_io.opensearch.config import (
    OpenSearchVectorIOConfig,
)
from llama_stack.providers.remote.vector_io.opensearch.opensearch import (
    OpenSearchVectorIOAdapter,
)


class TestOpenSearchVectorIOAdapter(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = OpenSearchVectorIOConfig(
            host="localhost",
            port=9200,
        )
        self.inference_api = MagicMock()
        self.files_api = MagicMock()

    @patch("llama_stack.providers.remote.vector_io.opensearch.opensearch.OpenSearch")
    async def test_initialize(self, mock_opensearch):
        adapter = OpenSearchVectorIOAdapter(self.config, self.inference_api, self.files_api)
        
        # Mock OpenSearch client
        mock_client = MagicMock()
        mock_opensearch.return_value = mock_client
        mock_client.info.return_value = {"version": {"number": "2.11.0"}}
        
        await adapter.initialize()
        
        mock_opensearch.assert_called_once()
        # Verify hosts config usage
        call_args = mock_opensearch.call_args[1]
        self.assertEqual(call_args["hosts"], [{"host": "localhost", "port": 9200}])

    @patch("llama_stack.providers.remote.vector_io.opensearch.opensearch.OpenSearch")
    async def test_register_vector_store(self, mock_opensearch):
        adapter = OpenSearchVectorIOAdapter(self.config, self.inference_api, self.files_api)
        mock_client = MagicMock()
        mock_opensearch.return_value = mock_client
        await adapter.initialize()

        vector_store = VectorStore(
            identifier="test_store",
            provider_id="test_provider",
            provider_resource_id="test_resource",
            embedding_dimension=384,
            embedding_model="test_model",
            status="completed",
        )

        mock_client.indices.exists.return_value = False
        
        await adapter.register_vector_store(vector_store)
        
        # Verify index creation
        mock_client.indices.create.assert_called_once()
        call_args = mock_client.indices.create.call_args[1]
        self.assertEqual(call_args["index"], "test_store")
        self.assertTrue("knn" in call_args["body"]["settings"]["index"])

    @patch("llama_stack.providers.remote.vector_io.opensearch.opensearch.OpenSearch")
    @patch("llama_stack.providers.remote.vector_io.opensearch.opensearch.helpers.bulk")
    async def test_insert_chunks(self, mock_bulk, mock_opensearch):
        adapter = OpenSearchVectorIOAdapter(self.config, self.inference_api, self.files_api)
        mock_client = MagicMock()
        mock_opensearch.return_value = mock_client
        mock_client.indices.exists.return_value = True
        
        await adapter.initialize()
        
        vector_store = VectorStore(
            identifier="test_store",
            provider_id="test_provider",
            provider_resource_id="test_resource",
            embedding_dimension=4,
            embedding_model="test_model",
            status="completed",
        )
        await adapter.register_vector_store(vector_store)

        chunks = [
            EmbeddedChunk(
                chunk_id="chunk1",
                content="test content",
                embedding=[0.1, 0.2, 0.3, 0.4],
                metadata={"key": "value"},
            )
        ]
        
        mock_bulk.return_value = (1, [])
        
        await adapter.insert_chunks(
            InsertChunksRequest(
                vector_store_id="test_store",
                chunks=chunks,
            )
        )
        
        mock_bulk.assert_called_once()
        self.assertEqual(mock_bulk.call_args[0][0], mock_client)
        actions = mock_bulk.call_args[0][1]
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["_index"], "test_store")
        self.assertEqual(actions[0]["_id"], "chunk1")

    @patch("llama_stack.providers.remote.vector_io.opensearch.opensearch.OpenSearch")
    async def test_query_chunks(self, mock_opensearch):
        adapter = OpenSearchVectorIOAdapter(self.config, self.inference_api, self.files_api)
        mock_client = MagicMock()
        mock_opensearch.return_value = mock_client
        mock_client.indices.exists.return_value = True
        
        await adapter.initialize()
        
        vector_store = VectorStore(
            identifier="test_store",
            provider_id="test_provider",
            provider_resource_id="test_resource",
            embedding_dimension=4,
            embedding_model="test_model",
            status="completed",
        )
        await adapter.register_vector_store(vector_store)

        # Mock search response
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_score": 0.9,
                        "_source": {
                            "chunk_id": "chunk1",
                            "content": "test content",
                            "embedding": [0.1, 0.2, 0.3, 0.4],
                            "metadata": {"key": "value"},
                            "chunk_content": None,
                        },
                    }
                ]
            }
        }

        response = await adapter.query_chunks(
            QueryChunksRequest(
                vector_store_id="test_store",
                query=[0.1, 0.2, 0.3, 0.4], # embedding query
                params={
                    "mode": "vector",
                    "score_threshold": 0.5,
                }
            )
        )
        
        self.assertEqual(len(response.chunks), 1)
        self.assertEqual(response.chunks[0].chunk_id, "chunk1")
        self.assertEqual(response.scores[0], 0.9)

if __name__ == "__main__":
    unittest.main()

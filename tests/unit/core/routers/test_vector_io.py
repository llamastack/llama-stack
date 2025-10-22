# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, Mock

import pytest

from llama_stack.apis.vector_io import OpenAICreateVectorStoreRequestWithExtraBody
from llama_stack.core.routers.vector_io import VectorIORouter


async def test_single_provider_auto_selection():
    # provider_id automatically selected during vector store create() when only one provider available
    mock_routing_table = Mock()
    mock_routing_table.impls_by_provider_id = {"inline::faiss": "mock_provider"}
    mock_routing_table.get_all_with_type = AsyncMock(
        return_value=[
            Mock(identifier="all-MiniLM-L6-v2", model_type="embedding", metadata={"embedding_dimension": 384})
        ]
    )
    mock_routing_table.register_vector_store = AsyncMock(
        return_value=Mock(identifier="vs_123", provider_id="inline::faiss", provider_resource_id="vs_123")
    )
    mock_routing_table.get_provider_impl = AsyncMock(
        return_value=Mock(openai_create_vector_store=AsyncMock(return_value=Mock(id="vs_123")))
    )
    router = VectorIORouter(mock_routing_table)
    request = OpenAICreateVectorStoreRequestWithExtraBody.model_validate(
        {"name": "test_store", "embedding_model": "all-MiniLM-L6-v2"}
    )

    result = await router.openai_create_vector_store(request)
    assert result.id == "vs_123"


async def test_create_vector_stores_multiple_providers_missing_provider_id_error():
    # if multiple providers are available, vector store create will error without provider_id
    mock_routing_table = Mock()
    mock_routing_table.impls_by_provider_id = {
        "inline::faiss": "mock_provider_1",
        "inline::sqlite-vec": "mock_provider_2",
    }
    mock_routing_table.get_all_with_type = AsyncMock(
        return_value=[
            Mock(identifier="all-MiniLM-L6-v2", model_type="embedding", metadata={"embedding_dimension": 384})
        ]
    )
    router = VectorIORouter(mock_routing_table)
    request = OpenAICreateVectorStoreRequestWithExtraBody.model_validate(
        {"name": "test_store", "embedding_model": "all-MiniLM-L6-v2"}
    )

    with pytest.raises(ValueError, match="Multiple vector_io providers available"):
        await router.openai_create_vector_store(request)

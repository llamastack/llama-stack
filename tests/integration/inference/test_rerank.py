# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from llama_stack_client import BadRequestError as LlamaStackBadRequestError
from llama_stack_client.types import RerankResponse
from llama_stack_client.types.shared.interleaved_content import (
    ImageContentItem,
    ImageContentItemImage,
    ImageContentItemImageURL,
    TextContentItem,
)

from llama_stack.core.library_client import LlamaStackAsLibraryClient

# Test data
DUMMY_STRING = "string_1"
DUMMY_STRING2 = "string_2"
DUMMY_TEXT = TextContentItem(text=DUMMY_STRING, type="text")
DUMMY_TEXT2 = TextContentItem(text=DUMMY_STRING2, type="text")
DUMMY_IMAGE_URL = ImageContentItem(
    image=ImageContentItemImage(url=ImageContentItemImageURL(uri="https://example.com/image.jpg")), type="image"
)
DUMMY_IMAGE_BASE64 = ImageContentItem(image=ImageContentItemImage(data="base64string"), type="image")

SUPPORTED_PROVIDERS = {"remote::nvidia"}
PROVIDERS_SUPPORTING_MEDIA = {}  # Providers that support media input for rerank models


def _validate_rerank_response(response: RerankResponse, items: list) -> None:
    """
    Validate that a rerank response has the correct structure and ordering.

    Args:
        response: The RerankResponse to validate
        items: The original items list that was ranked

    Raises:
        AssertionError: If any validation fails
    """
    seen = set()
    last_score = float("inf")
    for d in response.data:
        assert 0 <= d.index < len(items), f"Index {d.index} out of bounds for {len(items)} items"
        assert d.index not in seen, f"Duplicate index {d.index} found"
        seen.add(d.index)
        assert isinstance(d.relevance_score, float), f"Score must be float, got {type(d.relevance_score)}"
        assert d.relevance_score <= last_score, f"Scores not in descending order: {d.relevance_score} > {last_score}"
        last_score = d.relevance_score


@pytest.mark.parametrize(
    "query,items",
    [
        (DUMMY_STRING, [DUMMY_STRING, DUMMY_STRING2]),
        (DUMMY_TEXT, [DUMMY_TEXT, DUMMY_TEXT2]),
        (DUMMY_STRING, [DUMMY_STRING2, DUMMY_TEXT]),
        (DUMMY_TEXT, [DUMMY_STRING, DUMMY_TEXT2]),
    ],
    ids=[
        "string-query-string-items",
        "text-query-text-items",
        "mixed-content-1",
        "mixed-content-2",
    ],
)
def test_rerank_text(llama_stack_client, rerank_model_id, query, items, inference_provider_type):
    if inference_provider_type not in SUPPORTED_PROVIDERS:
        pytest.xfail(f"{inference_provider_type} doesn't support rerank models yet. ")

    response = llama_stack_client.inference.rerank(model=rerank_model_id, query=query, items=items)
    assert isinstance(response, RerankResponse)
    assert len(response.data) <= len(items)
    _validate_rerank_response(response, items)


@pytest.mark.parametrize(
    "query,items",
    [
        (DUMMY_IMAGE_URL, [DUMMY_STRING]),
        (DUMMY_IMAGE_BASE64, [DUMMY_TEXT]),
        (DUMMY_TEXT, [DUMMY_IMAGE_URL]),
        (DUMMY_IMAGE_BASE64, [DUMMY_IMAGE_URL, DUMMY_STRING, DUMMY_IMAGE_BASE64, DUMMY_TEXT]),
        (DUMMY_TEXT, [DUMMY_IMAGE_URL, DUMMY_STRING, DUMMY_IMAGE_BASE64, DUMMY_TEXT]),
    ],
    ids=[
        "image-query-url",
        "image-query-base64",
        "text-query-image-item",
        "mixed-content-1",
        "mixed-content-2",
    ],
)
def test_rerank_image(llama_stack_client, rerank_model_id, query, items, inference_provider_type):
    if inference_provider_type not in SUPPORTED_PROVIDERS:
        pytest.xfail(f"{inference_provider_type} doesn't support rerank models yet. ")

    if rerank_model_id not in PROVIDERS_SUPPORTING_MEDIA:
        error_type = (
            ValueError if isinstance(llama_stack_client, LlamaStackAsLibraryClient) else LlamaStackBadRequestError
        )
        with pytest.raises(error_type):
            llama_stack_client.inference.rerank(model=rerank_model_id, query=query, items=items)
    else:
        response = llama_stack_client.inference.rerank(model=rerank_model_id, query=query, items=items)

        assert isinstance(response, RerankResponse)
        assert len(response.data) <= len(items)
        _validate_rerank_response(response, items)


def test_rerank_max_results(llama_stack_client, rerank_model_id, inference_provider_type):
    if inference_provider_type not in SUPPORTED_PROVIDERS:
        pytest.xfail(f"{inference_provider_type} doesn't support rerank models yet. ")

    items = [DUMMY_STRING, DUMMY_STRING2, DUMMY_TEXT, DUMMY_TEXT2]
    max_num_results = 2

    response = llama_stack_client.inference.rerank(
        model=rerank_model_id,
        query=DUMMY_STRING,
        items=items,
        max_num_results=max_num_results,
    )

    assert isinstance(response, RerankResponse)
    assert len(response.data) == max_num_results
    _validate_rerank_response(response, items)


def test_rerank_max_results_larger_than_items(llama_stack_client, rerank_model_id, inference_provider_type):
    if inference_provider_type not in SUPPORTED_PROVIDERS:
        pytest.xfail(f"{inference_provider_type} doesn't support rerank yet")

    items = [DUMMY_STRING, DUMMY_STRING2]
    response = llama_stack_client.inference.rerank(
        model=rerank_model_id,
        query=DUMMY_STRING,
        items=items,
        max_num_results=10,  # Larger than items length
    )

    assert isinstance(response, RerankResponse)
    assert len(response.data) <= len(items)  # Should return at most len(items)

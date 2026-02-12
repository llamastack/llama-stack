# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import MagicMock, patch

import pytest

from llama_stack.providers.utils.inference.reranking_mixin import DEFAULT_RERANKER_INSTRUCTION, TransformerRerankerMixin
from llama_stack_api import (
    OpenAIChatCompletionContentPartImageParam,
    OpenAIChatCompletionContentPartTextParam,
)


class ConcreteReranker(TransformerRerankerMixin):
    def __init__(self):
        self.model_store = MagicMock()


class TestExtractText:
    def test_extract_text_from_string(self):
        reranker = ConcreteReranker()
        assert reranker._extract_text("hello world") == "hello world"

    def test_extract_text_from_text_content_part(self):
        reranker = ConcreteReranker()
        text_part = OpenAIChatCompletionContentPartTextParam(text="test text", type="text")
        assert reranker._extract_text(text_part) == "test text"

    def test_extract_text_from_image_raises(self):
        reranker = ConcreteReranker()
        image_part = MagicMock(spec=OpenAIChatCompletionContentPartImageParam)
        with pytest.raises(ValueError, match="Unsupported content type for reranking"):
            reranker._extract_text(image_part)


class TestFormatInstruction:
    def test_format_instruction_basic(self):
        reranker = ConcreteReranker()
        result = reranker._format_instruction("Retrieve relevant passages", "What is Python?", "Python is a language.")
        assert "<Instruct>: Retrieve relevant passages" in result
        assert "<Query>: What is Python?" in result
        assert "<Document>: Python is a language." in result

    def test_format_instruction_default_instruction(self):
        reranker = ConcreteReranker()
        result = reranker._format_instruction(DEFAULT_RERANKER_INSTRUCTION, "query", "doc")
        assert "<Instruct>: Given the search query, retrieve relevant passages that answer the query" in result


class TestRerank:
    async def test_rerank_empty_items_returns_empty(self):
        reranker = ConcreteReranker()
        result = await reranker.rerank(model="test-model", query="test query", items=[])
        assert result.data == []

    @pytest.mark.parametrize("max_num_results", [0, -1])
    async def test_rerank_invalid_max_num_results_raises(self, max_num_results):
        reranker = ConcreteReranker()
        with pytest.raises(ValueError, match="max_num_results must be >= 1"):
            await reranker.rerank(
                model="test-model",
                query="test query",
                items=["some item"],
                max_num_results=max_num_results,
            )

    @patch.object(TransformerRerankerMixin, "_load_reranker_model")
    @patch.object(TransformerRerankerMixin, "_compute_reranked_scores")
    async def test_rerank_returns_sorted_results(self, mock_compute, mock_load):
        reranker = ConcreteReranker()

        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_load.return_value = (mock_tokenizer, mock_model)
        mock_compute.return_value = [0.3, 0.1, 0.9]

        result = await reranker.rerank(
            model="test-model",
            query="test query",
            items=[
                "Python is a programming language",
                "The capital of Ireland is Dublin",
                "Machine learning is a subset of AI",
            ],
        )

        # Results should be sorted by score descending
        assert len(result.data) == 3
        assert result.data[0].index == 2
        assert result.data[0].relevance_score == pytest.approx(0.9)
        assert result.data[1].index == 0
        assert result.data[1].relevance_score == pytest.approx(0.3)
        assert result.data[2].index == 1
        assert result.data[2].relevance_score == pytest.approx(0.1)

    @patch.object(TransformerRerankerMixin, "_load_reranker_model")
    @patch.object(TransformerRerankerMixin, "_compute_reranked_scores")
    async def test_rerank_respects_max_num_results(self, mock_compute, mock_load):
        reranker = ConcreteReranker()

        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_load.return_value = (mock_tokenizer, mock_model)
        mock_compute.return_value = [0.5, 0.9, 0.1, 0.7]

        result = await reranker.rerank(
            model="test-model",
            query="test query",
            items=["a", "b", "c", "d"],
            max_num_results=2,
        )

        # Should only return top 2
        assert len(result.data) == 2
        assert result.data[0].index == 1  # score 0.9
        assert result.data[1].index == 3  # score 0.7

    @patch.object(TransformerRerankerMixin, "_load_reranker_model")
    @patch.object(TransformerRerankerMixin, "_compute_reranked_scores")
    async def test_rerank_builds_correct_pairs(self, mock_compute, mock_load):
        reranker = ConcreteReranker()

        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_load.return_value = (mock_tokenizer, mock_model)
        mock_compute.return_value = [0.5, 0.5]

        await reranker.rerank(
            model="test-model",
            query="What is AI?",
            items=["AI is intelligence", "Cars are fast"],
        )

        call_args = mock_compute.call_args[0]
        pairs = call_args[2]
        assert len(pairs) == 2
        assert "<Query>: What is AI?" in pairs[0]
        assert "<Document>: AI is intelligence" in pairs[0]
        assert "<Document>: Cars are fast" in pairs[1]

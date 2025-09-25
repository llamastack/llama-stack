# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import base64
import struct
from typing import TYPE_CHECKING

from llama_stack.log import get_logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from llama_stack.apis.inference import (
    EmbeddingsResponse,
    EmbeddingTaskType,
    InterleavedContentItem,
    ModelStore,
    OpenAIEmbeddingData,
    OpenAIEmbeddingsResponse,
    OpenAIEmbeddingUsage,
    TextTruncation,
)
from llama_stack.providers.utils.inference.prompt_adapter import interleaved_content_as_str

EMBEDDING_MODELS = {}


log = get_logger(name=__name__, category="providers::utils")


class SentenceTransformerEmbeddingMixin:
    model_store: ModelStore

    async def embeddings(
        self,
        model_id: str,
        contents: list[str] | list[InterleavedContentItem],
        text_truncation: TextTruncation | None = TextTruncation.none,
        output_dimension: int | None = None,
        task_type: EmbeddingTaskType | None = None,
    ) -> EmbeddingsResponse:
        model = await self.model_store.get_model(model_id)
        embedding_model = await self._load_sentence_transformer_model(model.provider_resource_id)
        embeddings = await asyncio.to_thread(
            embedding_model.encode,
            [interleaved_content_as_str(content) for content in contents],
            show_progress_bar=False,
        )
        return EmbeddingsResponse(embeddings=embeddings)

    async def openai_embeddings(
        self,
        model: str,
        input: str | list[str],
        encoding_format: str | None = "float",
        dimensions: int | None = None,
        user: str | None = None,
    ) -> OpenAIEmbeddingsResponse:
        # Convert input to list format if it's a single string
        input_list = [input] if isinstance(input, str) else input
        if not input_list:
            raise ValueError("Empty list not supported")

        # Get the model and generate embeddings
        model_obj = await self.model_store.get_model(model)
        embedding_model = await self._load_sentence_transformer_model(model_obj.provider_resource_id)
        embeddings = await asyncio.to_thread(embedding_model.encode, input_list, show_progress_bar=False)

        # Convert embeddings to the requested format
        data = []
        for i, embedding in enumerate(embeddings):
            if encoding_format == "base64":
                # Convert float array to base64 string
                float_bytes = struct.pack(f"{len(embedding)}f", *embedding)
                embedding_value = base64.b64encode(float_bytes).decode("ascii")
            else:
                # Default to float format
                embedding_value = embedding.tolist()

            data.append(
                OpenAIEmbeddingData(
                    embedding=embedding_value,
                    index=i,
                )
            )

        # Not returning actual token usage
        usage = OpenAIEmbeddingUsage(prompt_tokens=-1, total_tokens=-1)
        return OpenAIEmbeddingsResponse(
            data=data,
            model=model,
            usage=usage,
        )

    async def _load_sentence_transformer_model(self, model: str) -> "SentenceTransformer":
        global EMBEDDING_MODELS

        loaded_model = EMBEDDING_MODELS.get(model)
        if loaded_model is not None:
            return loaded_model

        log.info(f"Loading sentence transformer for {model}...")

        def _load_model():
            from sentence_transformers import SentenceTransformer

            return SentenceTransformer(model)

        loaded_model = await asyncio.to_thread(_load_model)
        EMBEDDING_MODELS[model] = loaded_model
        return loaded_model

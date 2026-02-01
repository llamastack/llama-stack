# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import base64
import platform
import struct
from typing import TYPE_CHECKING

import torch

from llama_stack.log import get_logger
from llama_stack.providers.utils.common.security_config import TrustedModelConfig

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from llama_stack_api import (
    ModelStore,
    OpenAIEmbeddingData,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
    OpenAIEmbeddingUsage,
    validate_embeddings_input_is_text,
)

EMBEDDING_MODELS: dict[str, "SentenceTransformer"] = {}
EMBEDDING_MODELS_LOCK = asyncio.Lock()

DARWIN = "Darwin"


log = get_logger(name=__name__, category="providers::utils")


class SentenceTransformerEmbeddingMixin:
    model_store: ModelStore
    trusted_model_config: TrustedModelConfig

    async def openai_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        # Validate that input contains only text, not token arrays
        validate_embeddings_input_is_text(params)

        # Convert input to list format if it's a single string
        input_list = [params.input] if isinstance(params.input, str) else params.input
        if not input_list:
            raise ValueError("Empty list not supported")

        # Get the model and generate embeddings
        embedding_model = await self._load_sentence_transformer_model(params.model)
        embeddings = await asyncio.to_thread(embedding_model.encode, input_list, show_progress_bar=False)

        # Convert embeddings to the requested format
        data = []
        for i, embedding in enumerate(embeddings):
            if params.encoding_format == "base64":
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
            model=params.model,
            usage=usage,
        )

    async def _load_sentence_transformer_model(self, model: str) -> "SentenceTransformer":
        loaded_model = EMBEDDING_MODELS.get(model)
        if loaded_model is not None:
            return loaded_model

        async with EMBEDDING_MODELS_LOCK:
            loaded_model = EMBEDDING_MODELS.get(model)
            if loaded_model is not None:
                return loaded_model

            trust_remote = self.trusted_model_config.is_trusted_model(model)

            if not trust_remote:
                log.warning(
                    f"Model {model} is not in trusted list. "
                    f"Loading with trust_remote_code=False for security. "
                    f"Trusted prefixes: {self.trusted_model_config.trusted_model_prefixes}"
                )
            else:
                log.info(f"Model {model} is in trusted list, loading with trust_remote_code=True")

            def _load_model():
                from sentence_transformers import SentenceTransformer

                platform_name = platform.system()
                if platform_name == DARWIN:
                    # PyTorch's OpenMP kernels can segfault on macOS when spawned from background
                    # threads with the default parallel settings, so force a single-threaded CPU run.
                    log.debug(f"Constraining torch threads on {platform_name} to a single worker")
                    torch.set_num_threads(1)
                return SentenceTransformer(model, trust_remote_code=trust_remote)

            loaded_model = await asyncio.to_thread(_load_model)
            EMBEDDING_MODELS[model] = loaded_model
            return loaded_model

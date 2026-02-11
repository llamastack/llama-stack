# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import io
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any

import chardet
import numpy as np
import tiktoken
from numpy.typing import NDArray
from pypdf import PdfReader

from llama_stack.core.datatypes import VectorStoresConfig
from llama_stack.log import get_logger
from llama_stack.providers.utils.inference.prompt_adapter import (
    interleaved_content_as_str,
)
from llama_stack.providers.utils.vector_io.vector_utils import generate_chunk_id
from llama_stack_api import (
    Chunk,
    ChunkForDeletion,
    ChunkMetadata,
    EmbeddedChunk,
    Inference,
    InsertChunksRequest,
    OpenAIEmbeddingsRequestWithExtraBody,
    QueryChunksRequest,
    QueryChunksResponse,
    VectorStore,
)

log = get_logger(name=__name__, category="providers::utils")


@cache
def _get_encoding(name: str) -> tiktoken.Encoding:
    return tiktoken.get_encoding(name)


# Constants for reranker types
RERANKER_TYPE_RRF = "rrf"
RERANKER_TYPE_WEIGHTED = "weighted"
RERANKER_TYPE_NORMALIZED = "normalized"


def parse_pdf(data: bytes) -> str:
    # For PDF and DOC/DOCX files, we can't reliably convert to string
    pdf_bytes = io.BytesIO(data)

    pdf_reader = PdfReader(pdf_bytes)
    return "\n".join([page.extract_text() for page in pdf_reader.pages])


def parse_data_url(data_url: str):
    data_url_pattern = re.compile(
        r"^"
        r"data:"
        r"(?P<mimetype>[\w/\-+.]+)"
        r"(?P<charset>;charset=(?P<encoding>[\w-]+))?"
        r"(?P<base64>;base64)?"
        r",(?P<data>.*)"
        r"$",
        re.DOTALL,
    )
    match = data_url_pattern.match(data_url)
    if not match:
        raise ValueError("Invalid Data URL format")

    parts = match.groupdict()
    parts["is_base64"] = bool(parts["base64"])
    return parts


def content_from_data_and_mime_type(data: bytes | str, mime_type: str | None, encoding: str | None = None) -> str:
    if isinstance(data, str):
        return data

    mime_category = mime_type.split("/")[0] if mime_type else None
    if mime_category == "text":
        if not encoding:
            detected = chardet.detect(data)
            encoding = detected["encoding"] or "utf-8"

        # For text-based files (including CSV, MD)
        try:
            return data.decode(encoding)
        except UnicodeDecodeError as e:
            log.warning(f"Decoding with encoding {encoding} failed: {e}")
            if encoding.lower() != "utf-8":
                try:
                    return data.decode("utf-8")
                except UnicodeDecodeError as e_utf8:
                    log.warning(f"Decoding with UTF-8 fallback also failed: {e_utf8}")
            raise e

    elif mime_type == "application/pdf":
        return parse_pdf(data)

    else:
        log.error("Could not extract content from data_url properly.")
        return ""


def make_overlapped_chunks(
    document_id: str,
    text: str,
    window_len: int,
    overlap_len: int,
    metadata: dict[str, Any],
    chunk_tokenizer_encoding: str = "cl100k_base",
) -> list[Chunk]:
    """Split `text` into overlapping windows while tracking the tokenizer used.

    The function converts the document and metadata into tokens via `tiktoken`,
    defaulting to `cl100k_base` but allowing callers to override the encoding name
    if they already know the embedding model's tokenizer. Each window advances by
    `window_len - overlap_len`, decodes for storage, and records the tokenizer name and token
    counts on both metadata layers so downstream consumers can stay within their limits.
    Downstream tokenizers may split the decoded text into more tokens than the original
    encoding, so adjust `window_len` when targeting a model whose tokenizer expands the
    chunk beyond its token limit."""
    encoding_name = chunk_tokenizer_encoding
    encoding = _get_encoding(encoding_name)
    chunk_tokenizer_name = f"tiktoken:{encoding_name}"
    tokens = encoding.encode(text)
    try:
        metadata_string = str(metadata)
    except Exception as e:
        raise ValueError("Failed to serialize metadata to string") from e

    metadata_tokens = encoding.encode(metadata_string)

    chunks = []
    for i in range(0, len(tokens), window_len - overlap_len):
        toks = tokens[i : i + window_len]
        chunk = encoding.decode(toks)
        chunk_window = f"{i}-{i + len(toks)}"
        chunk_id = generate_chunk_id(chunk, text, chunk_window)
        chunk_metadata = metadata.copy()
        chunk_metadata["chunk_id"] = chunk_id
        chunk_metadata["document_id"] = document_id
        chunk_metadata["token_count"] = len(toks)
        chunk_metadata["metadata_token_count"] = len(metadata_tokens)
        chunk_metadata["chunk_tokenizer"] = chunk_tokenizer_name

        backend_chunk_metadata = ChunkMetadata(
            chunk_id=chunk_id,
            document_id=document_id,
            source=metadata.get("source", None),
            created_timestamp=metadata.get("created_timestamp", int(time.time())),
            updated_timestamp=int(time.time()),
            chunk_window=chunk_window,
            chunk_tokenizer=chunk_tokenizer_name,
            content_token_count=len(toks),
            metadata_token_count=len(metadata_tokens),
        )

        # chunk is a string
        chunks.append(
            Chunk(
                content=chunk,
                chunk_id=chunk_id,
                metadata=chunk_metadata,
                chunk_metadata=backend_chunk_metadata,
            )
        )

    return chunks


type EmbeddingSequence = Sequence[float | int | np.number] | NDArray[Any]


def _validate_embedding(embedding: EmbeddingSequence, index: int, expected_dimension: int):
    """Helper method to validate embedding format and dimensions"""
    if not isinstance(embedding, (list | np.ndarray)):
        raise ValueError(f"Embedding at index {index} must be a list or numpy array, got {type(embedding)}")

    if isinstance(embedding, np.ndarray):
        if not np.issubdtype(embedding.dtype, np.number):
            raise ValueError(f"Embedding at index {index} contains non-numeric values")
    else:
        if not all(isinstance(e, (float | int | np.number)) for e in embedding):
            raise ValueError(f"Embedding at index {index} contains non-numeric values")

    if len(embedding) != expected_dimension:
        raise ValueError(f"Embedding at index {index} has dimension {len(embedding)}, expected {expected_dimension}")


class EmbeddingIndex(ABC):
    @abstractmethod
    async def add_chunks(self, embedded_chunks: list[EmbeddedChunk]):
        raise NotImplementedError()

    @abstractmethod
    async def delete_chunks(self, chunks_for_deletion: list[ChunkForDeletion]):
        raise NotImplementedError()

    @abstractmethod
    async def query_vector(self, embedding: NDArray, k: int, score_threshold: float) -> QueryChunksResponse:
        raise NotImplementedError()

    @abstractmethod
    async def query_keyword(self, query_string: str, k: int, score_threshold: float) -> QueryChunksResponse:
        raise NotImplementedError()

    @abstractmethod
    async def query_hybrid(
        self,
        embedding: NDArray,
        query_string: str,
        k: int,
        score_threshold: float,
        reranker_type: str,
        reranker_params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        raise NotImplementedError()

    @abstractmethod
    async def delete(self):
        raise NotImplementedError()


@dataclass
class VectorStoreWithIndex:
    vector_store: VectorStore
    index: EmbeddingIndex
    inference_api: Inference
    vector_stores_config: VectorStoresConfig | None = None

    async def insert_chunks(
        self,
        request: InsertChunksRequest,
    ) -> None:
        # Validate embedding dimensions match the vector store
        for i, embedded_chunk in enumerate(request.chunks):
            _validate_embedding(embedded_chunk.embedding, i, self.vector_store.embedding_dimension)

        await self.index.add_chunks(request.chunks)

    async def query_chunks(
        self,
        request: QueryChunksRequest,
    ) -> QueryChunksResponse:
        config = self.vector_stores_config or VectorStoresConfig()

        params = request.params
        if params is None:
            params = {}
        k = params.get("max_chunks", 3)
        mode = params.get("mode")
        score_threshold = params.get("score_threshold", 0.0)

        # Get reranker configuration from params (set by openai_vector_store_mixin)
        # NOTE: Breaking change - removed support for old nested "ranker" format.
        #       Now uses flattened format: reranker_type and reranker_params.
        reranker_type = params.get("reranker_type")
        reranker_params = params.get("reranker_params", {})

        # If no ranker specified, use VectorStoresConfig default
        if reranker_type is None:
            reranker_type = (
                RERANKER_TYPE_RRF
                if config.chunk_retrieval_params.default_reranker_strategy == "rrf"
                else config.chunk_retrieval_params.default_reranker_strategy
            )
            reranker_params = {"impact_factor": config.chunk_retrieval_params.rrf_impact_factor}

        # Normalize reranker_type to use constants
        if reranker_type == "weighted":
            reranker_type = RERANKER_TYPE_WEIGHTED
            # Ensure alpha is set (use default if not provided)
            if "alpha" not in reranker_params:
                reranker_params["alpha"] = config.chunk_retrieval_params.weighted_search_alpha
        elif reranker_type == "rrf":
            reranker_type = RERANKER_TYPE_RRF
            # Ensure impact_factor is set (use default if not provided)
            if "impact_factor" not in reranker_params:
                reranker_params["impact_factor"] = config.chunk_retrieval_params.rrf_impact_factor
        elif reranker_type == "neural":
            # TODO: Implement neural reranking
            log.warning(
                "TODO: Neural reranking for vector stores is not implemented yet; "
                "using configured reranker params without algorithm fallback."
            )
        elif reranker_type == "normalized":
            reranker_type = RERANKER_TYPE_NORMALIZED
        else:
            # Default to RRF for unknown strategies
            reranker_type = RERANKER_TYPE_RRF
            if "impact_factor" not in reranker_params:
                reranker_params["impact_factor"] = config.chunk_retrieval_params.rrf_impact_factor

        # Store neural model and weights from params if provided (for future neural reranking in Part II)
        if "neural_model" in params:
            reranker_params["neural_model"] = params["neural_model"]
        if "neural_weights" in params:
            reranker_params["neural_weights"] = params["neural_weights"]

        query_string = interleaved_content_as_str(request.query)
        if mode == "keyword":
            return await self.index.query_keyword(query_string, k, score_threshold)

        if "embedding_dimensions" in params:
            embeddings_request = OpenAIEmbeddingsRequestWithExtraBody(
                model=self.vector_store.embedding_model,
                input=[query_string],
                dimensions=params.get("embedding_dimensions"),
            )
        else:
            embeddings_request = OpenAIEmbeddingsRequestWithExtraBody(
                model=self.vector_store.embedding_model, input=[query_string]
            )
        embeddings_response = await self.inference_api.openai_embeddings(embeddings_request)
        query_vector = np.array(embeddings_response.data[0].embedding, dtype=np.float32)
        if mode == "hybrid":
            return await self.index.query_hybrid(
                query_vector, query_string, k, score_threshold, reranker_type, reranker_params
            )
        else:
            return await self.index.query_vector(query_vector, k, score_threshold)

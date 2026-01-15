# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import base64
import io
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote

import httpx
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from llama_stack.core.datatypes import VectorStoresConfig
from llama_stack.log import get_logger
from llama_stack.models.llama.llama3.tokenizer import Tokenizer
from llama_stack.providers.utils.inference.prompt_adapter import (
    interleaved_content_as_str,
)
from llama_stack.providers.utils.vector_io.vector_utils import generate_chunk_id
from llama_stack_api import (
    URL,
    Api,
    Chunk,
    ChunkMetadata,
    EmbeddedChunk,
    InterleavedContent,
    OpenAIEmbeddingsRequestWithExtraBody,
    QueryChunksResponse,
    RAGDocument,
    VectorStore,
)

log = get_logger(name=__name__, category="providers::utils")


class ChunkForDeletion(BaseModel):
    """Information needed to delete a chunk from a vector store.

    :param chunk_id: The ID of the chunk to delete
    :param document_id: The ID of the document this chunk belongs to
    """

    chunk_id: str
    document_id: str


# Constants for reranker types
RERANKER_TYPE_RRF = "rrf"
RERANKER_TYPE_WEIGHTED = "weighted"
RERANKER_TYPE_NORMALIZED = "normalized"


def parse_pdf(data: bytes) -> str:
    # For PDF and DOC/DOCX files, we can't reliably convert to string
    pdf_bytes = io.BytesIO(data)
    from pypdf import PdfReader

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


def content_from_data(data_url: str) -> str:
    parts = parse_data_url(data_url)
    data = parts["data"]

    if parts["is_base64"]:
        data = base64.b64decode(data)
    else:
        data = unquote(data)
        encoding = parts["encoding"] or "utf-8"
        data = data.encode(encoding)
    return content_from_data_and_mime_type(data, parts["mimetype"], parts.get("encoding", None))


def content_from_data_and_mime_type(data: bytes | str, mime_type: str | None, encoding: str | None = None) -> str:
    if isinstance(data, bytes):
        if not encoding:
            import chardet

            detected = chardet.detect(data)
            encoding = detected["encoding"]

    mime_category = mime_type.split("/")[0] if mime_type else None
    if mime_category == "text":
        # For text-based files (including CSV, MD)
        encodings_to_try = [encoding]
        if encoding != "utf-8":
            encodings_to_try.append("utf-8")
        first_exception = None
        for encoding in encodings_to_try:
            try:
                return data.decode(encoding)
            except UnicodeDecodeError as e:
                if first_exception is None:
                    first_exception = e
                log.warning(f"Decoding failed with {encoding}: {e}")
        # raise the origional exception, if we got here there was at least 1 exception
        log.error(f"Could not decode data as any of {encodings_to_try}")
        raise first_exception

    elif mime_type == "application/pdf":
        return parse_pdf(data)

    else:
        log.error("Could not extract content from data_url properly.")
        return ""


async def content_from_doc(doc: RAGDocument) -> str:
    if isinstance(doc.content, URL):
        uri = doc.content.uri
        if uri.startswith("file://"):
            raise ValueError("file:// URIs are not supported. Please use the Files API (/v1/files) to upload files.")
        if uri.startswith("data:"):
            return content_from_data(uri)
        async with httpx.AsyncClient() as client:
            r = await client.get(uri)
        if doc.mime_type == "application/pdf":
            return parse_pdf(r.content)
        return r.text
    elif isinstance(doc.content, str):
        if doc.content.startswith("file://"):
            raise ValueError("file:// URIs are not supported. Please use the Files API (/v1/files) to upload files.")
        pattern = re.compile("^(https?://|data:)")
        if pattern.match(doc.content):
            if doc.content.startswith("data:"):
                return content_from_data(doc.content)
            async with httpx.AsyncClient() as client:
                r = await client.get(doc.content)
            if doc.mime_type == "application/pdf":
                return parse_pdf(r.content)
            return r.text
        return doc.content
    else:
        # will raise ValueError if the content is not List[InterleavedContent] or InterleavedContent
        return interleaved_content_as_str(doc.content)


def make_overlapped_chunks(
    document_id: str,
    text: str,
    window_len: int,
    overlap_len: int,
    metadata: dict[str, Any],
) -> list[Chunk]:
    default_tokenizer = "DEFAULT_TIKTOKEN_TOKENIZER"
    tokenizer = Tokenizer.get_instance()
    tokens = tokenizer.encode(text, bos=False, eos=False)
    try:
        metadata_string = str(metadata)
    except Exception as e:
        raise ValueError("Failed to serialize metadata to string") from e

    metadata_tokens = tokenizer.encode(metadata_string, bos=False, eos=False)

    chunks = []
    for i in range(0, len(tokens), window_len - overlap_len):
        toks = tokens[i : i + window_len]
        chunk = tokenizer.decode(toks)
        chunk_window = f"{i}-{i + len(toks)}"
        chunk_id = generate_chunk_id(chunk, text, chunk_window)
        chunk_metadata = metadata.copy()
        chunk_metadata["chunk_id"] = chunk_id
        chunk_metadata["document_id"] = document_id
        chunk_metadata["token_count"] = len(toks)
        chunk_metadata["metadata_token_count"] = len(metadata_tokens)

        backend_chunk_metadata = ChunkMetadata(
            chunk_id=chunk_id,
            document_id=document_id,
            source=metadata.get("source", None),
            created_timestamp=metadata.get("created_timestamp", int(time.time())),
            updated_timestamp=int(time.time()),
            chunk_window=chunk_window,
            chunk_tokenizer=default_tokenizer,
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


def _validate_embedding(embedding: NDArray, index: int, expected_dimension: int):
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
    inference_api: Api.inference
    vector_stores_config: VectorStoresConfig | None = None

    async def insert_chunks(
        self,
        chunks: list[EmbeddedChunk],
    ) -> None:
        # Validate embedding dimensions match the vector store
        for i, embedded_chunk in enumerate(chunks):
            _validate_embedding(embedded_chunk.embedding, i, self.vector_store.embedding_dimension)

        await self.index.add_chunks(chunks)

    async def query_chunks(
        self,
        query: InterleavedContent,
        params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        config = self.vector_stores_config or VectorStoresConfig()

        if params is None:
            params = {}
        k = params.get("max_chunks", 3)
        mode = params.get("mode")
        score_threshold = params.get("score_threshold", 0.0)

        ranker = params.get("ranker")
        if ranker is None:
            reranker_type = (
                RERANKER_TYPE_RRF
                if config.chunk_retrieval_params.default_reranker_strategy == "rrf"
                else config.chunk_retrieval_params.default_reranker_strategy
            )
            reranker_params = {"impact_factor": config.chunk_retrieval_params.rrf_impact_factor}
        else:
            strategy = ranker.get("strategy", config.chunk_retrieval_params.default_reranker_strategy)
            if strategy == "weighted":
                weights = ranker.get("params", {}).get("weights", [0.5, 0.5])
                reranker_type = RERANKER_TYPE_WEIGHTED
                reranker_params = {
                    "alpha": weights[0] if len(weights) > 0 else config.chunk_retrieval_params.weighted_search_alpha
                }
            elif strategy == "normalized":
                reranker_type = RERANKER_TYPE_NORMALIZED
            else:
                reranker_type = RERANKER_TYPE_RRF
                k_value = ranker.get("params", {}).get("k", config.chunk_retrieval_params.rrf_impact_factor)
                reranker_params = {"impact_factor": k_value}

        query_string = interleaved_content_as_str(query)
        if mode == "keyword":
            return await self.index.query_keyword(query_string, k, score_threshold)

        params = OpenAIEmbeddingsRequestWithExtraBody(
            model=self.vector_store.embedding_model,
            input=[query_string],
        )
        embeddings_response = await self.inference_api.openai_embeddings(params)
        query_vector = np.array(embeddings_response.data[0].embedding, dtype=np.float32)
        if mode == "hybrid":
            return await self.index.query_hybrid(
                query_vector, query_string, k, score_threshold, reranker_type, reranker_params
            )
        else:
            return await self.index.query_vector(query_vector, k, score_threshold)

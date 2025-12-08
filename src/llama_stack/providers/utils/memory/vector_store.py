# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import asyncio
import base64
import io
import mimetypes
import os
import re
import stat
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import httpx
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

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

# Maximum file size for file:// URIs (default 100MB, configurable via env)
MAX_FILE_URI_SIZE_BYTES = int(os.environ.get("LLAMA_STACK_MAX_FILE_URI_SIZE_MB", "100")) * 1024 * 1024
ALLOW_FILE_URI = os.environ.get("LLAMA_STACK_ALLOW_FILE_URI", "false").lower() in ("true", "1", "yes")


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


async def read_file_uri(uri: str, max_size: int | None = None) -> tuple[bytes, str | None]:
    parsed = urlparse(uri)
    file_path = unquote(parsed.path)
    real_path = os.path.realpath(file_path)
    filename = os.path.basename(real_path)

    file_stat = os.stat(real_path)
    if stat.S_ISDIR(file_stat.st_mode):
        raise IsADirectoryError(f"Cannot read directory: {filename}")
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"Not a regular file: {filename}")

    file_size = file_stat.st_size
    size_limit = max_size if max_size is not None else MAX_FILE_URI_SIZE_BYTES
    if file_size > size_limit:
        raise ValueError(f"File too large: {file_size} bytes exceeds limit of {size_limit} bytes")

    content = await asyncio.to_thread(Path(real_path).read_bytes)
    mime_type, _ = mimetypes.guess_type(real_path)
    return content, mime_type


async def content_from_doc(doc: RAGDocument) -> str:
    if isinstance(doc.content, URL):
        uri = doc.content.uri
    elif isinstance(doc.content, str):
        uri = doc.content
    else:
        return interleaved_content_as_str(doc.content)

    if uri.startswith("data:"):
        return content_from_data(uri)

    if uri.startswith("file://"):
        if not ALLOW_FILE_URI:
            raise ValueError(
                "file:// URIs disabled. Use Files API (/v1/files) instead, or set LLAMA_STACK_ALLOW_FILE_URI=true."
            )
        content, guessed_mime = await read_file_uri(uri)
        mime = doc.mime_type or guessed_mime
        return parse_pdf(content) if mime == "application/pdf" else content.decode("utf-8")

    if uri.startswith("http://") or uri.startswith("https://"):
        async with httpx.AsyncClient() as client:
            r = await client.get(uri)
        if doc.mime_type == "application/pdf":
            return parse_pdf(r.content)
        return r.text

    if isinstance(doc.content, str):
        return doc.content

    raise ValueError(f"Unsupported URL scheme: {uri}")


def make_overlapped_chunks(
    document_id: str, text: str, window_len: int, overlap_len: int, metadata: dict[str, Any]
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
            chunk_embedding_model=None,  # This will be set in `VectorStoreWithIndex.insert_chunks`
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
    async def add_chunks(self, chunks: list[Chunk], embeddings: NDArray):
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

    async def insert_chunks(
        self,
        chunks: list[Chunk],
    ) -> None:
        chunks_to_embed = []
        for i, c in enumerate(chunks):
            if c.embedding is None:
                chunks_to_embed.append(c)
                if c.chunk_metadata:
                    c.chunk_metadata.chunk_embedding_model = self.vector_store.embedding_model
                    c.chunk_metadata.chunk_embedding_dimension = self.vector_store.embedding_dimension
            else:
                _validate_embedding(c.embedding, i, self.vector_store.embedding_dimension)

        if chunks_to_embed:
            params = OpenAIEmbeddingsRequestWithExtraBody(
                model=self.vector_store.embedding_model,
                input=[c.content for c in chunks_to_embed],
            )
            resp = await self.inference_api.openai_embeddings(params)
            for c, data in zip(chunks_to_embed, resp.data, strict=False):
                c.embedding = data.embedding

        embeddings = np.array([c.embedding for c in chunks], dtype=np.float32)
        await self.index.add_chunks(chunks, embeddings)

    async def query_chunks(
        self,
        query: InterleavedContent,
        params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        if params is None:
            params = {}
        k = params.get("max_chunks", 3)
        mode = params.get("mode")
        score_threshold = params.get("score_threshold", 0.0)

        ranker = params.get("ranker")
        if ranker is None:
            reranker_type = RERANKER_TYPE_RRF
            reranker_params = {"impact_factor": 60.0}
        else:
            strategy = ranker.get("strategy", "rrf")
            if strategy == "weighted":
                weights = ranker.get("params", {}).get("weights", [0.5, 0.5])
                reranker_type = RERANKER_TYPE_WEIGHTED
                reranker_params = {"alpha": weights[0] if len(weights) > 0 else 0.5}
            elif strategy == "normalized":
                reranker_type = RERANKER_TYPE_NORMALIZED
            else:
                reranker_type = RERANKER_TYPE_RRF
                k_value = ranker.get("params", {}).get("k", 60.0)
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

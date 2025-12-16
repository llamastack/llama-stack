# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Default prompt template for query rewriting in vector search
DEFAULT_QUERY_REWRITE_PROMPT = "Expand this query with relevant synonyms and related terms. Return only the improved query, no explanations:\n\n{query}\n\nImproved query:"

# Default templates for file search tool output formatting
DEFAULT_FILE_SEARCH_HEADER_TEMPLATE = (
    "knowledge_search tool found {num_chunks} chunks:\nBEGIN of knowledge_search tool results.\n"
)
DEFAULT_FILE_SEARCH_FOOTER_TEMPLATE = "END of knowledge_search tool results.\n"

# Default templates for LLM prompt content and chunk formatting
DEFAULT_CHUNK_ANNOTATION_TEMPLATE = "Result {index}\nContent: {chunk.content}\nMetadata: {metadata}\n"
DEFAULT_CONTEXT_TEMPLATE = 'The above results were retrieved to help answer the user\'s query: "{query}". Use them as supporting information only in answering this query.{annotation_instruction}\n'

# Default templates for source annotation and attribution features
DEFAULT_ANNOTATION_INSTRUCTION_TEMPLATE = " Cite sources immediately at the end of sentences before punctuation, using `<|file-id|>` format like 'This is a fact <|file-Cn3MSNn72ENTiiq11Qda4A|>.'. Do not add extra punctuation. Use only the file IDs provided, do not invent new ones."
DEFAULT_CHUNK_WITH_SOURCES_TEMPLATE = "[{index}] {metadata_text} cite as <|{file_id}|>\n{chunk_text}\n"

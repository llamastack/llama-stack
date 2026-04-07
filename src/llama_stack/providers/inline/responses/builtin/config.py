# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from llama_stack.core.datatypes import VectorStoresConfig
from llama_stack.core.storage.datatypes import KVStoreReference, ResponsesStoreReference

DEFAULT_SUMMARIZATION_PROMPT = (
    "You are performing a CONTEXT CHECKPOINT COMPACTION. Create a concise handoff summary "
    "of the conversation so far. Include:\n"
    "- Current progress and key decisions made\n"
    "- Important context, constraints, or user preferences\n"
    "- What remains to be done (clear next steps)\n"
    "- Any critical data, examples, or references needed to continue\n\n"
    "Be concise, structured, and focused on helping seamlessly continue the work."
)


class CompactionConfig(BaseModel):
    """Configuration for conversation compaction behavior and prompt templates."""

    summarization_prompt: str = Field(
        default=DEFAULT_SUMMARIZATION_PROMPT,
        description="Prompt template used to instruct the model to summarize conversation history during compaction.",
    )
    summarization_model: str | None = Field(
        default=None,
        description="Model to use for generating compaction summaries. If not set, uses the same model as the conversation.",
    )
    default_compact_threshold: int | None = Field(
        default=None,
        description="Default token threshold for auto-compaction via context_management. If set, conversations exceeding this token count will be automatically compacted.",
    )
    tokenizer_encoding: str | None = Field(
        default=None,
        description="Tiktoken encoding name for token counting (e.g. 'o200k_base', 'cl100k_base'). If not set, the encoding is resolved from the model name via tiktoken.encoding_for_model().",
    )


class ResponsesPersistenceConfig(BaseModel):
    """Nested persistence configuration for the responses provider."""

    agent_state: KVStoreReference
    responses: ResponsesStoreReference


class BuiltinResponsesImplConfig(BaseModel):
    """Configuration for the built-in responses with persistence and vector store settings."""

    persistence: ResponsesPersistenceConfig

    vector_stores_config: VectorStoresConfig | None = Field(
        default=None,
        description="Configuration for vector store prompt templates and behavior",
    )

    compaction_config: CompactionConfig = Field(
        default_factory=CompactionConfig,
        description="Configuration for conversation compaction behavior and prompt templates",
    )

    @classmethod
    def sample_run_config(cls, __distro_dir__: str) -> dict[str, Any]:
        return {
            "persistence": {
                "agent_state": KVStoreReference(
                    backend="kv_default",
                    namespace="agents",
                ).model_dump(exclude_none=True),
                "responses": ResponsesStoreReference(
                    backend="sql_default",
                    table_name="responses",
                ).model_dump(exclude_none=True),
            }
        }

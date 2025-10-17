# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Central definition of integration test suites. You can use these suites by passing --suite=name to pytest.
# For example:
#
# ```bash
# pytest tests/integration/ --suite=vision --setup=ollama
# ```
#
"""
Each suite defines what to run (roots). Suites can be run with different global setups defined in setups.py.
Setups provide environment variables and model defaults that can be reused across multiple suites.

CLI examples:
  pytest tests/integration --suite=responses --setup=gpt
  pytest tests/integration --suite=vision --setup=ollama
  pytest tests/integration --suite=base --setup=vllm
"""

from pathlib import Path

from pydantic import BaseModel, Field

this_dir = Path(__file__).parent


class Suite(BaseModel):
    name: str
    roots: list[str]
    default_setup: str | None = None


class Setup(BaseModel):
    """A reusable test configuration with environment and CLI defaults."""

    name: str
    description: str
    defaults: dict[str, str | int] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)


# Global setups - can be used with any suite "technically" but in reality, some setups might work
# only for specific test suites.
SETUP_DEFINITIONS: dict[str, Setup] = {
    "ollama": Setup(
        name="ollama",
        description="Local Ollama provider with text + safety models",
        env={
            "OLLAMA_URL": "http://0.0.0.0:11434",
            "SAFETY_MODEL": "ollama/llama-guard3:1b",
        },
        defaults={
            "text_model": "ollama/llama3.2:3b-instruct-fp16",
            "embedding_model": "ollama/nomic-embed-text:v1.5",
            "safety_model": "ollama/llama-guard3:1b",
            "safety_shield": "llama-guard",
        },
    ),
    "ollama-vision": Setup(
        name="ollama",
        description="Local Ollama provider with a vision model",
        env={
            "OLLAMA_URL": "http://0.0.0.0:11434",
        },
        defaults={
            "vision_model": "ollama/llama3.2-vision:11b",
            "embedding_model": "ollama/nomic-embed-text:v1.5",
        },
    ),
    "vllm": Setup(
        name="vllm",
        description="vLLM provider with a text model",
        env={
            "VLLM_URL": "http://localhost:8000/v1",
        },
        defaults={
            "text_model": "vllm/meta-llama/Llama-3.2-1B-Instruct",
            "embedding_model": "sentence-transformers/nomic-embed-text-v1.5",
        },
    ),
    "gpt": Setup(
        name="gpt",
        description="OpenAI GPT models for high-quality responses and tool calling",
        defaults={
            "text_model": "openai/gpt-4o",
            "embedding_model": "openai/text-embedding-3-small",
            "embedding_dimension": 1536,
        },
    ),
    "tgi": Setup(
        name="tgi",
        description="Text Generation Inference (TGI) provider with a text model",
        env={
            "TGI_URL": "http://localhost:8080",
        },
        defaults={
            "text_model": "tgi/Qwen/Qwen3-0.6B",
        },
    ),
    "together": Setup(
        name="together",
        description="Together computer models",
        defaults={
            "text_model": "together/meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            "embedding_model": "together/togethercomputer/m2-bert-80M-32k-retrieval",
        },
    ),
    "cerebras": Setup(
        name="cerebras",
        description="Cerebras models",
        defaults={
            "text_model": "cerebras/llama-3.3-70b",
        },
    ),
    "databricks": Setup(
        name="databricks",
        description="Databricks models",
        defaults={
            "text_model": "databricks/databricks-meta-llama-3-3-70b-instruct",
            "embedding_model": "databricks/databricks-bge-large-en",
        },
    ),
    "fireworks": Setup(
        name="fireworks",
        description="Fireworks provider with a text model",
        defaults={
            "text_model": "fireworks/accounts/fireworks/models/llama-v3p1-8b-instruct",
            "embedding_model": "fireworks/accounts/fireworks/models/qwen3-embedding-8b",
        },
    ),
    "anthropic": Setup(
        name="anthropic",
        description="Anthropic Claude models",
        defaults={
            "text_model": "anthropic/claude-3-5-haiku-20241022",
        },
    ),
    "llama-api": Setup(
        name="llama-openai-compat",
        description="Llama models from https://api.llama.com",
        defaults={
            "text_model": "llama_openai_compat/Llama-3.3-8B-Instruct",
        },
    ),
    "groq": Setup(
        name="groq",
        description="Groq models",
        defaults={
            "text_model": "groq/llama-3.3-70b-versatile",
        },
    ),
    "milvus": Setup(
        name="milvus",
        description="Milvus vector database provider for vector_io tests",
        env={
            "MILVUS_URL": "dummy",
        },
    ),
    "chromadb": Setup(
        name="chromadb",
        description="ChromaDB vector database provider for vector_io tests",
        env={
            "CHROMADB_URL": "http://localhost:8000",
        },
    ),
    "pgvector": Setup(
        name="pgvector",
        description="PGVector database provider for vector_io tests",
        env={
            "PGVECTOR_DB": "llama_stack_test",
            "PGVECTOR_USER": "postgres",
            "PGVECTOR_PASSWORD": "password",
        },
    ),
}


base_roots = [
    str(p)
    for p in this_dir.glob("*")
    if p.is_dir()
    and p.name not in ("__pycache__", "fixtures", "test_cases", "recordings", "responses", "post_training")
]

SUITE_DEFINITIONS: dict[str, Suite] = {
    "base": Suite(
        name="base",
        roots=base_roots,
        default_setup="ollama",
    ),
    "responses": Suite(
        name="responses",
        roots=["tests/integration/responses"],
        default_setup="gpt",
    ),
    "vision": Suite(
        name="vision",
        roots=["tests/integration/inference/test_vision_inference.py"],
        default_setup="ollama-vision",
    ),
    "vector_io": Suite(
        name="vector_io",
        roots=["tests/integration/vector_io"],
        default_setup="milvus",
    ),
}

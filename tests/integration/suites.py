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
    defaults: dict[str, str] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)


# Global setups - can be used with any suite "technically" but in reality, some setups might work
# only for specific test suites.
SETUP_DEFINITIONS: dict[str, Setup] = {
    "ollama": Setup(
        name="ollama",
        description="Local Ollama provider with text + safety models",
        env={
            "OLLAMA_URL": "http://0.0.0.0:11434",
        },
        defaults={
            "text_model": "ollama/llama3.2:3b-instruct-fp16",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
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
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
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
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        },
    ),
    "gpt": Setup(
        name="gpt",
        description="OpenAI GPT models for high-quality responses and tool calling",
        defaults={
            "text_model": "openai/gpt-4o",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
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
}

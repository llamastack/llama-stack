# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Central definition of integration test suites. You can use these suites by passing --suite=name to pytest.
# For example:
#
# ```bash
# pytest tests/integration/ --suite=vision
# ```
#
# Each suite can:
# - restrict collection to specific roots (dirs or files)
# - provide default CLI option values (e.g. text_model, embedding_model, etc.)

from pathlib import Path

this_dir = Path(__file__).parent
default_roots = [
    str(p)
    for p in this_dir.glob("*")
    if p.is_dir()
    and p.name not in ("__pycache__", "fixtures", "test_cases", "recordings", "responses", "post_training")
]

SUITE_DEFINITIONS: dict[str, dict] = {
    "base": {
        "description": "Base suite that includes most tests but runs them with a text Ollama model",
        "roots": default_roots,
        "defaults": {
            "text_model": "ollama/llama3.2:3b-instruct-fp16",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        },
    },
    "responses": {
        "description": "Suite that includes only the OpenAI Responses tests; needs a strong tool-calling model",
        "roots": ["tests/integration/responses"],
        "defaults": {
            "text_model": "openai/gpt-4o",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        },
    },
    "vision": {
        "description": "Suite that includes only the vision tests",
        "roots": ["tests/integration/inference/test_vision_inference.py"],
        "defaults": {
            "vision_model": "ollama/llama3.2-vision:11b",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        },
    },
}

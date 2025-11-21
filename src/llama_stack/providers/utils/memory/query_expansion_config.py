# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.core.datatypes import QualifiedModel, VectorStoresConfig
from llama_stack.providers.utils.memory.constants import DEFAULT_QUERY_EXPANSION_PROMPT

# Global configuration for query expansion - set during stack startup
_DEFAULT_QUERY_EXPANSION_MODEL: QualifiedModel | None = None
_DEFAULT_QUERY_EXPANSION_MAX_TOKENS: int = 100
_DEFAULT_QUERY_EXPANSION_TEMPERATURE: float = 0.3
_QUERY_EXPANSION_PROMPT_OVERRIDE: str | None = None


def set_default_query_expansion_config(vector_stores_config: VectorStoresConfig | None):
    """Set the global default query expansion configuration from stack config."""
    global \
        _DEFAULT_QUERY_EXPANSION_MODEL, \
        _QUERY_EXPANSION_PROMPT_OVERRIDE, \
        _DEFAULT_QUERY_EXPANSION_MAX_TOKENS, \
        _DEFAULT_QUERY_EXPANSION_TEMPERATURE
    if vector_stores_config:
        _DEFAULT_QUERY_EXPANSION_MODEL = vector_stores_config.default_query_expansion_model
        # Only set override if user provided a custom prompt different from default
        if vector_stores_config.query_expansion_prompt != DEFAULT_QUERY_EXPANSION_PROMPT:
            _QUERY_EXPANSION_PROMPT_OVERRIDE = vector_stores_config.query_expansion_prompt
        else:
            _QUERY_EXPANSION_PROMPT_OVERRIDE = None
        _DEFAULT_QUERY_EXPANSION_MAX_TOKENS = vector_stores_config.query_expansion_max_tokens
        _DEFAULT_QUERY_EXPANSION_TEMPERATURE = vector_stores_config.query_expansion_temperature
    else:
        _DEFAULT_QUERY_EXPANSION_MODEL = None
        _QUERY_EXPANSION_PROMPT_OVERRIDE = None
        _DEFAULT_QUERY_EXPANSION_MAX_TOKENS = 100
        _DEFAULT_QUERY_EXPANSION_TEMPERATURE = 0.3

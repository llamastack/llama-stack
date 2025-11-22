# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.core.datatypes import QualifiedModel, VectorStoresConfig
from llama_stack.providers.utils.memory.constants import DEFAULT_QUERY_EXPANSION_PROMPT

# Global configuration for query rewriting - set during stack startup
_DEFAULT_REWRITE_QUERY_MODEL: QualifiedModel | None = None
_DEFAULT_REWRITE_QUERY_MAX_TOKENS: int = 100
_DEFAULT_REWRITE_QUERY_TEMPERATURE: float = 0.3
_REWRITE_QUERY_PROMPT_OVERRIDE: str | None = None


def set_default_rewrite_query_config(vector_stores_config: VectorStoresConfig | None):
    """Set the global default query rewriting configuration from stack config."""
    global \
        _DEFAULT_REWRITE_QUERY_MODEL, \
        _REWRITE_QUERY_PROMPT_OVERRIDE, \
        _DEFAULT_REWRITE_QUERY_MAX_TOKENS, \
        _DEFAULT_REWRITE_QUERY_TEMPERATURE
    if vector_stores_config and vector_stores_config.rewrite_query_params:
        params = vector_stores_config.rewrite_query_params
        _DEFAULT_REWRITE_QUERY_MODEL = params.model
        # Only set override if user provided a custom prompt different from default
        if params.prompt != DEFAULT_QUERY_EXPANSION_PROMPT:
            _REWRITE_QUERY_PROMPT_OVERRIDE = params.prompt
        else:
            _REWRITE_QUERY_PROMPT_OVERRIDE = None
        _DEFAULT_REWRITE_QUERY_MAX_TOKENS = params.max_tokens
        _DEFAULT_REWRITE_QUERY_TEMPERATURE = params.temperature
    else:
        _DEFAULT_REWRITE_QUERY_MODEL = None
        _REWRITE_QUERY_PROMPT_OVERRIDE = None
        _DEFAULT_REWRITE_QUERY_MAX_TOKENS = 100
        _DEFAULT_REWRITE_QUERY_TEMPERATURE = 0.3

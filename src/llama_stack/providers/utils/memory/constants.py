# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Default prompt template for query rewriting in vector search
DEFAULT_QUERY_REWRITE_PROMPT = "Expand this query with relevant synonyms and related terms. Return only the improved query, no explanations:\n\n{query}\n\nImproved query:"

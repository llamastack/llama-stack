# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.providers.datatypes import (
    AdapterSpec,
    Api,
    InlineProviderSpec,
    ProviderSpec,
    remote_provider_spec,
)


def available_providers() -> list[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.tool_runtime,
            provider_type="inline::rag-runtime",
            pip_packages=[
                "chardet",
                "pypdf",
                "tqdm",
                "numpy",
                "scikit-learn",
                "scipy",
                "nltk",
                "sentencepiece",
                "transformers",
            ],
            module="llama_stack.providers.inline.tool_runtime.rag",
            config_class="llama_stack.providers.inline.tool_runtime.rag.config.RagToolRuntimeConfig",
            api_dependencies=[Api.vector_io, Api.inference, Api.files],
            description="RAG (Retrieval-Augmented Generation) tool runtime for document ingestion, chunking, and semantic search.",
        ),
        remote_provider_spec(
            api=Api.tool_runtime,
            adapter=AdapterSpec(
                adapter_type="brave-search",
                module="llama_stack.providers.remote.tool_runtime.brave_search",
                config_class="llama_stack.providers.remote.tool_runtime.brave_search.config.BraveSearchToolConfig",
                pip_packages=["requests"],
                provider_data_validator="llama_stack.providers.remote.tool_runtime.brave_search.BraveSearchToolProviderDataValidator",
                description="Brave Search tool for web search capabilities with privacy-focused results.",
            ),
        ),
        remote_provider_spec(
            api=Api.tool_runtime,
            adapter=AdapterSpec(
                adapter_type="bing-search",
                module="llama_stack.providers.remote.tool_runtime.bing_search",
                config_class="llama_stack.providers.remote.tool_runtime.bing_search.config.BingSearchToolConfig",
                pip_packages=["requests"],
                provider_data_validator="llama_stack.providers.remote.tool_runtime.bing_search.BingSearchToolProviderDataValidator",
                description="Bing Search tool for web search capabilities using Microsoft's search engine.",
            ),
        ),
        remote_provider_spec(
            api=Api.tool_runtime,
            adapter=AdapterSpec(
                adapter_type="tavily-search",
                module="llama_stack.providers.remote.tool_runtime.tavily_search",
                config_class="llama_stack.providers.remote.tool_runtime.tavily_search.config.TavilySearchToolConfig",
                pip_packages=["requests"],
                provider_data_validator="llama_stack.providers.remote.tool_runtime.tavily_search.TavilySearchToolProviderDataValidator",
                description="Tavily Search tool for AI-optimized web search with structured results.",
            ),
        ),
        remote_provider_spec(
            api=Api.tool_runtime,
            adapter=AdapterSpec(
                adapter_type="wolfram-alpha",
                module="llama_stack.providers.remote.tool_runtime.wolfram_alpha",
                config_class="llama_stack.providers.remote.tool_runtime.wolfram_alpha.config.WolframAlphaToolConfig",
                pip_packages=["requests"],
                provider_data_validator="llama_stack.providers.remote.tool_runtime.wolfram_alpha.WolframAlphaToolProviderDataValidator",
                description="Wolfram Alpha tool for computational knowledge and mathematical calculations.",
            ),
        ),
        remote_provider_spec(
            api=Api.tool_runtime,
            adapter=AdapterSpec(
                adapter_type="model-context-protocol",
                module="llama_stack.providers.remote.tool_runtime.model_context_protocol",
                config_class="llama_stack.providers.remote.tool_runtime.model_context_protocol.config.MCPProviderConfig",
                pip_packages=["mcp>=1.8.1"],
                provider_data_validator="llama_stack.providers.remote.tool_runtime.model_context_protocol.config.MCPProviderDataValidator",
                description="Model Context Protocol (MCP) tool for standardized tool calling and context management.",
            ),
        ),
    ]

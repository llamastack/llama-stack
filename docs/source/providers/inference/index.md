# Inference

## Overview

Llama Stack Inference API for generating completions, chat completions, and embeddings.

    This API provides the raw interface to the underlying models. Two kinds of models are supported:
    - LLM models: these models generate "raw" and "chat" (conversational) completions.
    - Embedding models: these models generate embeddings to be used for semantic search.

This section contains documentation for all available providers for the **inference** API.

## Providers

```{toctree}
:maxdepth: 1

inline_meta-reference
inline_sentence-transformers
remote_anthropic
remote_azure
remote_bedrock
remote_cerebras
remote_databricks
remote_fireworks
remote_gemini
remote_groq
remote_hf_endpoint
remote_hf_serverless
remote_llama-openai-compat
remote_nvidia
remote_ollama
remote_openai
remote_passthrough
remote_runpod
remote_sambanova
remote_tgi
remote_together
remote_vertexai
remote_vllm
remote_watsonx
```

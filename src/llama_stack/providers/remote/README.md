# remote providers

Remote provider adapters that connect Llama Stack APIs to external services.

## Directory Structure

```
remote/
  inference/           # Remote inference adapters
    anthropic/         # Anthropic Claude
    azure/             # Azure OpenAI
    bedrock/           # AWS Bedrock
    cerebras/          # Cerebras Cloud
    databricks/        # Databricks
    fireworks/         # Fireworks AI
    gemini/            # Google Gemini
    groq/              # Groq
    llama_openai_compat/ # Generic OpenAI-compatible endpoints
    nvidia/            # NVIDIA NIM
    oci/               # Oracle Cloud Infrastructure
    ollama/            # Ollama
    openai/            # OpenAI
    passthrough/       # Generic passthrough to any endpoint
    runpod/            # RunPod
    sambanova/         # SambaNova
    tgi/               # HuggingFace TGI / Inference API
    together/          # Together AI
    vertexai/          # Google Vertex AI
    vllm/              # vLLM
    watsonx/           # IBM WatsonX
  agents/              # Remote agent services
  safety/              # Remote safety services
  vector_io/           # Remote vector storage (e.g., chromadb, qdrant, weaviate)
  datasetio/           # Remote dataset services
  eval/                # Remote evaluation services
  files/               # Remote file storage
  tool_runtime/        # Remote tool runtimes
  post_training/       # Remote training services
  __init__.py
```

## What Makes a Provider "Remote"

Remote providers adapt an external service to the Llama Stack API. They are declared with `RemoteProviderSpec` and their `provider_type` starts with `remote::` (e.g., `remote::ollama`).

Their factory function is typically named `get_adapter_impl()` and returns an instance implementing the relevant protocol from `llama_stack_api`.

## Common Pattern

Most remote inference providers extend `OpenAIMixin` from `providers/utils/inference/openai_mixin.py`, which provides a standard implementation of OpenAI-compatible chat completion, completion, and embedding endpoints using the `AsyncOpenAI` client. The provider only needs to supply the base URL and any provider-specific configuration.

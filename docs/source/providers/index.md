# API Providers

The goal of Llama Stack is to build an ecosystem where users can easily swap out different implementations for the same API. Examples for these include:
- LLM inference providers (e.g., Meta Reference, Ollama, Fireworks, Together, AWS Bedrock, Groq, Cerebras, SambaNova, vLLM, OpenAI, Anthropic, Gemini, WatsonX, etc.),
- Vector databases (e.g., FAISS, SQLite-Vec, ChromaDB, Weaviate, Qdrant, Milvus, PGVector, etc.),
- Safety providers (e.g., Meta's Llama Guard, Prompt Guard, Code Scanner, AWS Bedrock Guardrails, etc.),
- Tool Runtime providers (e.g., RAG Runtime, Brave Search, etc.)

Providers come in two flavors:
- **Remote**: the provider runs as a separate service external to the Llama Stack codebase. Llama Stack contains a small amount of adapter code.
- **Inline**: the provider is fully specified and implemented within the Llama Stack codebase. It may be a simple wrapper around an existing library, or a full fledged implementation within Llama Stack.

Importantly, Llama Stack always strives to provide at least one fully inline provider for each API so you can iterate on a fully featured environment locally.

```{toctree}
:maxdepth: 1

external/index
openai
inference/index
agents/index
datasetio/index
safety/index
telemetry/index
vector_io/index
tool_runtime/index
files/index
```

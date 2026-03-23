# inline providers

In-process provider implementations that run locally within the Llama Stack server.

## Directory Structure

```
inline/
  agents/              # Agent orchestration (meta-reference: responses API, tool calling)
  batches/             # Batch processing for async job execution
  inference/           # Local inference (meta-reference GPU, sentence-transformers)
  safety/              # Safety checks (llama-guard, code-scanner)
  vector_io/           # Vector storage (sqlite-vec, faiss, chroma, milvus, qdrant)
  datasetio/           # Dataset I/O (local file handling)
  eval/                # Evaluation orchestration
  scoring/             # Scoring function implementations
  tool_runtime/        # Tool runtime (RAG context retrieval)
  files/               # File storage and management
  file_processor/      # File processing (text extraction, etc.)
  post_training/       # Post-training / fine-tuning
  __init__.py
```

## What Makes a Provider "Inline"

Inline providers run in the same process as the server. They are declared with `InlineProviderSpec` and their `provider_type` starts with `inline::` (e.g., `inline::meta-reference`).

Their factory function is typically named `get_provider_impl()` and returns an instance implementing the relevant protocol from `llama_stack_api`.

## Key Inline Providers

- **`agents/meta_reference`** -- Implements the Agents and OpenAI Responses APIs. Handles tool calling loops, multi-turn conversations, and streaming.
- **`inference/meta_reference`** -- Runs Llama models locally on GPU using PyTorch.
- **`inference/sentence_transformers`** -- Runs embedding models using the sentence-transformers library.
- **`vector_io/sqlite_vec`** -- SQLite-based vector storage using the sqlite-vec extension.

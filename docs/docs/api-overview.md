# API Reference Overview

Llama Stack implements the OpenAI API and organizes endpoints by stability level. Use any OpenAI-compatible client to access these APIs.

## Stable APIs

**Production-ready, OpenAI-compatible endpoints.**

| API | Endpoint | Description |
|-----|----------|-------------|
| Chat Completions | `/v1/chat/completions` | Text and vision inference, streaming, tool calling |
| Completions | `/v1/completions` | Text completions |
| Embeddings | `/v1/embeddings` | Text embeddings |
| Models | `/v1/models` | Model listing and management |
| Files | `/v1/files` | File upload and management |
| Vector Stores | `/v1/vector_stores` | Document storage and semantic search |
| Batches | `/v1/batches` | Offline batch processing |
| Moderations | `/v1/moderations` | Content safety via Llama Guard |
| Responses | `/v1/responses` | Server-side agentic orchestration with tool calling, MCP, and file search |
| Conversations | `/v1/conversations` | Conversation state management |

These APIs follow semantic versioning and maintain backward compatibility within major versions.

[**Browse Stable APIs →**](/docs/api/)

## Experimental APIs

**Preview APIs that may change before becoming stable.**

Experimental endpoints (v1alpha, v1beta) include new capabilities under active development. They are feature-complete but may change based on feedback.

[**Browse Experimental APIs →**](/docs/api-experimental/)

## Deprecated APIs

**Legacy APIs scheduled for removal.**

See [deprecated APIs](/docs/api-deprecated/) for migration guidance.

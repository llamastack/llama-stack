# Llama Stack Release Notes

This file provides a summary of each release. For detailed release notes with migration instructions, code examples, and full contributor credits, see the linked documents.

<!-- New releases go here, at the top. See docs/releases/ for detailed notes. -->

## [0.5.0](docs/releases/RELEASE_NOTES_0.5.md) - 2026-02-05

Release 0.5 brings significant improvements to API consistency, OpenAI conformance, provider capabilities, and a major architectural refactoring of all APIs to use FastAPI routers.

### Highlights

- **Connectors API** for managing MCP server connections ([#4263](https://github.com/meta-llama/llama-stack/pull/4263))
- **Unified network configuration** with TLS/mTLS, proxy, and timeout support for all remote providers ([#4748](https://github.com/meta-llama/llama-stack/pull/4748))
- **Endpoint authorization** via YAML-based access control ([#4448](https://github.com/meta-llama/llama-stack/pull/4448))
- **Rerankers** for hybrid search in vector stores ([#4456](https://github.com/meta-llama/llama-stack/pull/4456))
- **Response API enhancements** including `reasoning.effort`, `max_output_tokens`, and `parallel_tool_calls`
- **New providers**: Elasticsearch and OCI 26ai vector stores
- **PGVector improvements**: HNSW/IVFFlat indexes, configurable distance metrics
- **FastAPI router migration** across all APIs for better OpenAPI docs and validation
- **ARM64 container image** support ([#4474](https://github.com/meta-llama/llama-stack/pull/4474))

### Breaking Changes

| Change | Type | PR |
|--------|------|-----|
| Post-Training API endpoints restructured (path params) | Hard | [#4606](https://github.com/meta-llama/llama-stack/pull/4606) |
| Embeddings API rejects explicit `null` for optional fields | Hard | [#4644](https://github.com/meta-llama/llama-stack/pull/4644) |
| Safety API provider interface changed to request objects | Hard | [#4643](https://github.com/meta-llama/llama-stack/pull/4643) |
| Meta-Reference GPU inference provider removed | Hard | [#4828](https://github.com/meta-llama/llama-stack/pull/4828) |
| Scope-based endpoint authorization removed | Hard | [#4734](https://github.com/meta-llama/llama-stack/pull/4734) |
| `image_name` renamed to `distro_name` | Deprecated | [#4396](https://github.com/meta-llama/llama-stack/pull/4396) |
| Eval API calling convention uses request objects | Deprecated | [#4425](https://github.com/meta-llama/llama-stack/pull/4425) |
| vLLM `tls_verify` moved to `network.tls.verify` | Deprecated | [#4748](https://github.com/meta-llama/llama-stack/pull/4748) |

See the [full release notes](docs/releases/RELEASE_NOTES_0.5.md) for migration instructions and detailed upgrade guide.

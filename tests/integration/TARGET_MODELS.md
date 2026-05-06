# Target Model Matrix

This document makes explicit the models and CI lanes that are already configured in `suites.py` and `ci_matrix.json`. It exists so reviewers and release engineers can see, per provider, what we intentionally test.

All data below is derived from:

- `tests/integration/suites.py` — SETUP_DEFINITIONS and SUITE_DEFINITIONS
- `tests/integration/ci_matrix.json` — which suite/setup combinations run in CI
- `docs/docs/api-openai/provider_matrix.md` — current Responses API coverage

## CI Lanes (Default)

These 16 jobs run on every PR via `ci_matrix.json`:

| Suite | Setup | Notes |
|-------|-------|-------|
| `base` | `ollama` | |
| `bedrock` | `bedrock` | library client only |
| `base` | `ollama-postgres` | server client only, Postgres store |
| `vision` | `ollama-vision` | |
| `responses` | `gpt` | Reference: 133/133 tests (100%) |
| `responses` | `azure` | 114/133 tests (86%) |
| `gpt-reasoning` | `gpt-reasoning` | 2 reasoning effort tests |
| `responses` | `watsonx` | Many internal skips (42% effective) |
| `responses` | `vertexai` | 73/133 tests (55%) |
| `bedrock-responses` | `bedrock` | Curated 6-file subset (27/133, 20%) |
| `base-vllm-subset` | `vllm` | Inference tests only |
| `vllm-reasoning` | `vllm` | `test_reasoning.py` only |
| `ollama-reasoning` | `ollama-reasoning` | |
| `messages` | `ollama-reasoning` | Messages API translation path |
| `messages-openai` | `gpt` | Messages API via OpenAI (translation codepath) |
| `interactions` | `gemini` | Gemini Interactions API |

Weekly scheduled jobs (Sunday 00:01 UTC):

| Suite | Setup |
|-------|-------|
| `base` | `vllm` |
| `base` | `vllm-qwen3next` |

## Target Models Per Provider

### Tier 1 — Full Coverage

| Setup | Text Model | Vision Model | Embedding Model |
|-------|-----------|-------------|----------------|
| `gpt` | `openai/gpt-4o` | `openai/gpt-4o` | `openai/text-embedding-3-small` |
| `gpt-reasoning` | `openai/o4-mini` | — | — |
| `azure` | `azure/gpt-4o` | `azure/gpt-4o` | `sentence-transformers/nomic-ai/nomic-embed-text-v1.5` |

### Tier 2 — Strategic Coverage

| Setup | Text Model | Vision Model | Embedding Model |
|-------|-----------|-------------|----------------|
| `bedrock` | `bedrock/openai.gpt-oss-20b-1:0` | — | `sentence-transformers/nomic-ai/nomic-embed-text-v1.5` |
| `vertexai` | `vertexai/publishers/google/models/gemini-2.0-flash` | same | `sentence-transformers/nomic-ai/nomic-embed-text-v1.5` |
| `watsonx` | `watsonx/meta-llama/llama-3-3-70b-instruct` | — | — |
| `vllm` | `vllm/Qwen/Qwen3-0.6B` | — | `sentence-transformers/nomic-embed-text-v1.5` |
| `ollama` | `ollama/llama3.2:3b-instruct-fp16` | — | `ollama/nomic-embed-text:v1.5` |
| `ollama-vision` | — | `ollama/llama3.2-vision:11b` | `ollama/nomic-embed-text:v1.5` |
| `ollama-reasoning` | `ollama/gpt-oss:20b` | — | — |

### Tier 3 — Supplementary

| Setup | Text Model | Embedding Model |
|-------|-----------|----------------|
| `gemini` | `gemini/gemini-2.5-flash-lite` | `gemini/text-embedding-004` |
| `anthropic` | `anthropic/claude-3-5-haiku-20241022` | — |
| `groq` | `groq/llama-3.3-70b-versatile` | — |
| `fireworks` | `fireworks/.../llama-v3p1-8b-instruct` | `fireworks/.../qwen3-embedding-8b` |
| `databricks` | `databricks/databricks-meta-llama-3-3-70b-instruct` | `databricks/databricks-bge-large-en` |

## Known Capability Gaps

These are provider-imposed limitations, not recording gaps. Expanding coverage for these providers requires model upgrades or provider-side improvements.

| Provider | Constraint | Evidence |
|----------|-----------|---------|
| WatsonX | Tool calling skipped: "WatsonX does not reliably support tool calling" | `test_tool_responses.py:41-47` |
| WatsonX | Structured output skipped: "WatsonX model does not reliably produce valid structured JSON output" | `test_structured_output.py` |
| WatsonX | Compact: recordings not available yet (unverified, not confirmed blocked) | `test_compact_responses.py:14-19` |
| Bedrock | `bedrock-responses` is a curated 6-file subset, not the full `responses` suite | `suites.py` SUITE_DEFINITIONS |
| Bedrock | Excluded from full suite: structured output, parallel tool calls, multi-turn tool tests | `suites.py` comment on `bedrock-responses` |
| vLLM | Qwen3-0.6B is too small for tool calling, structured output, or most responses features | Model size limitation |
| Ollama | Strong inference coverage via `base`/`vision`/`reasoning`/`messages` suites; no responses suite | Local model capability limitation |

## Responses Coverage Summary

From `docs/docs/api-openai/provider_matrix.md`:

| Provider | Tested | Passing | Coverage |
|----------|--------|---------|----------|
| OpenAI | 133 | 133 | 100% |
| Azure | 114 | 114 | 86% |
| Vertex AI | 73 | 73 | 55% |
| WatsonX | 56 | 56 | 42% |
| Bedrock | 27 | 27 | 20% |
| vLLM | 3 | 3 | 2% |
| Ollama | 2 | 2 | 2% |

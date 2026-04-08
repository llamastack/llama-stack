# Google Interactions API Front-End — Design Spec

**Date:** 2026-04-08
**Jira:** RHAISTRAT-1348
**Status:** Draft
**Scope:** Minimal (scope A) — core `POST /interactions` with text I/O and SSE streaming

## Motivation

Agents built against Google's Interactions API (ADK, Gemini ecosystem) should be able
to call Llama Stack using their native protocol without code changes. This is a front-end
API translation layer, not a backend provider — identical to how Llama Stack already
serves the Anthropic Messages API.

## Architecture

The implementation mirrors the existing `messages` (Anthropic) pattern:

```
Google Interactions request
  |
  v
FastAPI Router (POST /v1/interactions)
  |
  v
BuiltinInteractionsImpl (Interactions Protocol)
  |-- Translate Google -> OpenAI Chat Completions
  |-- Call inference_api.openai_chat_completion()
  |-- Translate OpenAI response -> Google Interactions format
  |
  v
Google Interactions response (JSON or SSE stream)
```

No native passthrough in v1 — no inference provider natively speaks the Interactions
protocol today. Can be added later following the Messages passthrough pattern.

## API Surface

### Endpoint

```
POST /v1/interactions
```

### Request Model — `GoogleCreateInteractionRequest`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | `str` | yes | Model identifier |
| `input` | `str \| list[GoogleInputItem]` | yes | Prompt string or conversation turns |
| `system_instruction` | `str \| None` | no | System prompt |
| `generation_config` | `GoogleGenerationConfig \| None` | no | Sampling parameters |
| `stream` | `bool` | no | Enable SSE streaming (default `false`) |
| `response_modalities` | `list[str] \| None` | no | Accepted for compat, ignored in v1 |

**Out of scope for v1:** `tools`, `previous_interaction_id`, `background`, `store`,
`response_format`, multi-modal input (images, audio, video).

### Input Formats

**String input:**
```json
{"input": "Tell me a joke"}
```

**Conversation turns:**
```json
{
  "input": [
    {"role": "user", "content": [{"type": "text", "text": "Question 1"}]},
    {"role": "model", "content": [{"type": "text", "text": "Answer 1"}]},
    {"role": "user", "content": [{"type": "text", "text": "Question 2"}]}
  ]
}
```

### Generation Config — `GoogleGenerationConfig`

| Field | Type | Description |
|-------|------|-------------|
| `temperature` | `float \| None` | Sampling temperature |
| `top_k` | `int \| None` | Top-k sampling |
| `top_p` | `float \| None` | Nucleus sampling |
| `max_output_tokens` | `int \| None` | Max tokens to generate |

### Response Model — `GoogleInteractionResponse`

```json
{
  "id": "interaction-abc123",
  "status": "completed",
  "outputs": [
    {"type": "text", "text": "Response content here"}
  ],
  "usage": {
    "input_tokens": 100,
    "output_tokens": 50,
    "total_tokens": 150
  }
}
```

### Streaming Events (SSE)

Triggered by `stream: true` in request body. Wire format: `event: <type>\ndata: <json>\n\n`

| Event | Payload |
|-------|---------|
| `interaction.start` | `{event_type, id, status}` |
| `content.start` | `{event_type, index, type: "text"}` |
| `content.delta` | `{event_type, index, delta: {type: "text", text: "..."}}` |
| `content.stop` | `{event_type, index}` |
| `interaction.complete` | `{event_type, id, status, usage}` |

### Error Response

Google error format:
```json
{
  "error": {
    "code": 400,
    "message": "Invalid request"
  }
}
```

## Translation Logic

### Request: Google -> OpenAI

| Google Interactions | OpenAI Chat Completions |
|---------------------|-------------------------|
| `model` | `model` |
| `input` (string) | `messages: [{role: "user", content: input}]` |
| `input` (turns) | `messages` — role `"model"` mapped to `"assistant"` |
| `system_instruction` | `messages: [{role: "system", content: ...}]` prepended |
| `generation_config.temperature` | `temperature` |
| `generation_config.top_p` | `top_p` |
| `generation_config.top_k` | extra_body `top_k` |
| `generation_config.max_output_tokens` | `max_tokens` |
| `stream` | `stream` |

### Response: OpenAI -> Google

| OpenAI Chat Completion | Google Interaction |
|------------------------|-------------------|
| `choices[0].message.content` | `outputs: [{type: "text", text: ...}]` |
| `choices[0].finish_reason` | `status: "completed"` |
| `usage.prompt_tokens` | `usage.input_tokens` |
| `usage.completion_tokens` | `usage.output_tokens` |
| sum | `usage.total_tokens` |
| `id` | `id` (prefixed `interaction-`) |

### Streaming: OpenAI chunks -> Google SSE

1. First chunk -> `interaction.start`
2. First `delta.content` -> `content.start` (index 0, type "text")
3. Each `delta.content` -> `content.delta`
4. `finish_reason` received -> `content.stop`, then `interaction.complete` with usage

## File Layout

### New files

```
src/llama_stack_api/interactions/
  __init__.py              # exports
  api.py                   # Interactions protocol
  models.py                # Pydantic models
  fastapi_routes.py        # POST /v1/interactions router

src/llama_stack/providers/inline/interactions/
  __init__.py              # get_provider_impl()
  config.py                # InteractionsConfig
  impl.py                  # BuiltinInteractionsImpl

src/llama_stack/providers/registry/interactions.py
```

### Registration touchpoints

1. `src/llama_stack_api/datatypes.py` — add `interactions = "interactions"` to `Api` enum
2. `src/llama_stack_api/__init__.py` — export `Interactions` protocol
3. `src/llama_stack/core/resolver.py` — map `Api.interactions` -> `Interactions`
4. Distribution configs — add `interactions` provider to starter
5. Run `scripts/distro_codegen.py` and `scripts/provider_codegen.py`

## Testing

### Unit tests — `tests/unit/test_interactions.py`

**Request translation:**
- String input -> single user message
- Conversation turns with `"model"` role -> `"assistant"`
- `system_instruction` -> system message prepended
- `generation_config` fields mapped correctly
- `top_k` passed via extra_body

**Response translation:**
- Text response -> `outputs: [{type: "text", text: ...}]`
- Usage mapping (prompt_tokens -> input_tokens, total computed)
- ID prefixed with `interaction-`

**Streaming translation:**
- Event ordering: `interaction.start` -> `content.start` -> `content.delta`(s) -> `content.stop` -> `interaction.complete`
- Usage included in `interaction.complete`

**Edge cases:**
- Empty input string
- Empty model response
- Missing usage in OpenAI response

### No integration tests for v1

No backend natively speaks this protocol. Manual validation with ADK client
documented in PR description.

## Future Extensions (out of scope)

- Tool calling (`function_call` output type, `function_result` input type)
- `previous_interaction_id` for server-side conversation state
- `GET /v1/interactions/{id}` retrieval endpoint
- `background: true` for long-running tasks
- Multi-modal input/output (images, audio)
- Native passthrough for providers that add Interactions support
- `response_format` for structured/JSON output

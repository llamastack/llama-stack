# AWS Bedrock Test Recording Guide

This guide explains how to record and update test fixtures for the AWS Bedrock provider.

## Overview

Bedrock integration tests use a record/replay mechanism. Tests run against pre-recorded API responses in CI (replay mode), eliminating the need for AWS credentials. Contributors with AWS access can re-record tests when needed.

## Prerequisites

1. **AWS Account** with access to Amazon Bedrock
2. **Model Access**: Request access to `openai.gpt-oss-20b-1:0` in us-west-2 region via AWS Console
3. **AWS CLI** configured with valid credentials

## Step 1: Generate Bearer Token

Bedrock provides short-term API keys via the AWS Console (expires in 12 hours).

### Using AWS Console

1. Go to [Amazon Bedrock Console](https://console.aws.amazon.com/bedrock/) (ensure you're in **us-west-2** region)
2. In the left sidebar under **Discover**, click **API keys**
3. Click **Generate short-term API keys**
4. Copy the generated API key (starts with `bedrock-api-key-...`)

The console provides the export command:
```bash
export AWS_BEARER_TOKEN_BEDROCK=bedrock-api-key-YmVkcm9jay5hbWF6b25hd3MuY29...
```

> **Important**: Copy the key immediately - you won't be able to retrieve it after closing the dialog.

## Step 2: Create Local Stack Configuration

Create `bedrock-test.yaml` in the repo root (for local recording only - not committed):

```yaml
version: 2
image_name: bedrock-test
apis:
  - inference
  - models
providers:
  inference:
    - provider_id: bedrock
      provider_type: remote::bedrock
      config:
        api_key: ${env.AWS_BEARER_TOKEN_BEDROCK:=replay-mode-dummy-key}
        region_name: us-west-2
storage:
  backends:
    kv_default:
      type: kv_sqlite
      db_path: ${env.SQLITE_STORE_DIR:=./.llama}/kvstore.db
    sql_default:
      type: sql_sqlite
      db_path: ${env.SQLITE_STORE_DIR:=./.llama}/sql_store.db
  stores:
    metadata:
      namespace: registry
      backend: kv_default
    inference:
      table_name: inference_store
      backend: sql_default
      max_write_queue_size: 10000
      num_writers: 4
    conversations:
      table_name: openai_conversations
      backend: sql_default
    prompts:
      namespace: prompts
      backend: kv_default
registered_resources:
  models:
    - model_id: openai.gpt-oss-20b-1:0
      provider_id: bedrock
      provider_resource_id: openai.gpt-oss-20b-1:0
      model_type: llm
      metadata:
        description: "OpenAI GPT-OSS 20B on Bedrock (us-west-2)"
  shields: []
  vector_dbs: []
  datasets: []
  scoring_fns: []
  benchmarks: []
  tool_groups: []
server:
  port: 8321
```

## Step 3: Set Environment Variables

```bash
export AWS_BEARER_TOKEN_BEDROCK="bedrock-api-key-<your-presigned-url>"
export AWS_DEFAULT_REGION=us-west-2
export SQLITE_STORE_DIR=$(mktemp -d)
```

## Step 4: Run Tests in Record Mode

**Important**: You must run from the `tests/integration/inference` directory for pytest to recognize the custom arguments.

```bash
cd tests/integration/inference

uv run pytest -v -s \
  test_openai_completion.py::test_openai_chat_completion_non_streaming \
  test_openai_completion.py::test_openai_chat_completion_streaming \
  test_openai_completion.py::test_inference_store \
  --setup=bedrock \
  --stack-config=../../../bedrock-test.yaml \
  --inference-mode=record \
  -k "client_with_models"
```

Expected: 6 tests pass (2 non-streaming, 2 streaming, 2 inference_store).

## Step 5: Verify Recordings Work

**Important**: Run from the `tests/integration/inference` directory.

```bash
# Test without credentials (replay mode)
unset AWS_BEARER_TOKEN_BEDROCK

cd tests/integration/inference

# Option 1: Use CI config (recommended - matches what CI runs)
uv run pytest -v -s \
  test_openai_completion.py::test_openai_chat_completion_non_streaming \
  test_openai_completion.py::test_openai_chat_completion_streaming \
  test_openai_completion.py::test_inference_store \
  --setup=bedrock \
  --stack-config=ci-tests::config.yaml \
  --inference-mode=replay \
  -k "client_with_models"

# Option 2: Use local config (for development)
uv run pytest -v -s \
  test_openai_completion.py::test_openai_chat_completion_non_streaming \
  test_openai_completion.py::test_openai_chat_completion_streaming \
  test_openai_completion.py::test_inference_store \
  --setup=bedrock \
  --stack-config=../../../bedrock-test.yaml \
  --inference-mode=replay \
  -k "client_with_models"
```

Expected: All 6 tests pass using recorded responses (no live API calls).

## Step 6: Commit New Recordings

Commit the new/updated JSON files in `tests/integration/inference/recordings/`.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `unrecognized arguments` | Run from `tests/integration/inference` directory |
| `UnknownOperationException` | Ensure base URL includes `/v1` |
| `Bearer Token has expired` | Generate a new token (12-hour expiry) |
| `API key not provided` | Check `AWS_BEARER_TOKEN_BEDROCK` env var |
| `Model not found` | Verify model access in AWS Bedrock Console |
| Tests skipped | Check provider not in skip list in test file |

## Recording File Format

Each recording is a JSON file in `tests/integration/inference/recordings/`:

```json
{
  "test_id": "tests/integration/.../test_name[params]",
  "request": {
    "method": "POST",
    "url": "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1/chat/completions",
    "body": {"model": "...", "messages": [...], "stream": false}
  },
  "response": {
    "body": {"__type__": "openai.types.chat...", "__data__": {...}},
    "is_streaming": false
  }
}
```

The filename is a hash of the request parameters, ensuring deterministic matching during replay.

## Bedrock API Limitations

| Feature | Supported | Notes |
|---------|-----------|-------|
| `/v1/chat/completions` | Yes | Both streaming and non-streaming modes work. |
| `/v1/completions` | No | `BedrockInferenceAdapter` raises `NotImplementedError`. Bedrock's OpenAI-compatible API does not support legacy completions. |
| `/v1/models` | No | Provider returns empty list. Models must be pre-registered in stack config. |
| `/v1/embeddings` | No | `BedrockInferenceAdapter` raises `NotImplementedError`. Use a different embedding provider. |
| Streaming | Yes | SSE streaming works correctly with `/v1/chat/completions`. |
| Tool calling | No | Bedrock's OpenAI-compatible endpoint does not support the `tools` parameter. Requests with tools return 400. |
| System messages | Yes | System role messages are processed correctly. |
| Multi-turn context | Yes | Conversation history is maintained across turns. |

## Model Information

- **Model**: `openai.gpt-oss-20b-1:0` (OpenAI GPT-OSS 20B)
- **Region**: us-west-2 only
- **Endpoint**: `https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1`

## Adding New Tests to Bedrock CI

To add a new test to the Bedrock CI suite:

### 1. Record the Test

```bash
cd tests/integration/inference

uv run pytest -v -s \
  test_openai_completion.py::your_new_test_function \
  --setup=bedrock \
  --stack-config=../../../bedrock-test.yaml \
  --inference-mode=record \
  -k "client_with_models"
```

### 2. Verify Replay Works

```bash
uv run pytest -v -s \
  test_openai_completion.py::your_new_test_function \
  --setup=bedrock \
  --stack-config=ci-tests::config.yaml \
  --inference-mode=replay \
  -k "client_with_models"
```

### 3. Add to Bedrock Suite

Edit `tests/integration/suites.py` and add your test to the bedrock suite roots:

```
"bedrock": Suite(
    name="bedrock",
    roots=[
        "tests/integration/inference/test_openai_completion.py::test_openai_chat_completion_non_streaming",
        "tests/integration/inference/test_openai_completion.py::test_openai_chat_completion_streaming",
        "tests/integration/inference/test_openai_completion.py::test_inference_store",
        "tests/integration/inference/test_openai_completion.py::your_new_test_function",  # ADD HERE
    ],
    default_setup="bedrock",
)
```

### 4. Commit Everything

Commit the new recordings and updated `suites.py`.

## CI Configuration

The Bedrock CI uses a dedicated suite that runs only recorded tests:

| File | Purpose |
|------|---------|
| `src/llama_stack/distributions/ci-tests/config.yaml` | Stack config with dummy API key and pre-registered model |
| `tests/integration/suites.py` | Defines `bedrock` suite with specific test paths |
| `tests/integration/ci_matrix.json` | CI entry with `allowed_clients: ["library"]` to run only in library mode |

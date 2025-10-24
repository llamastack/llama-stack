# Integration Testing Guide

Integration tests verify complete workflows across different providers using Llama Stack's record-replay system.

## Quick Start

```bash
# Run all integration tests with existing recordings
uv run --group test \
  pytest -sv tests/integration/ --stack-config=starter
```

## Configuration Options

You can see all options with:
```bash
cd tests/integration

# this will show a long list of options, look for "Custom options:"
pytest --help
```

Here are the most important options:
- `--stack-config`: specify the stack config to use. You have four ways to point to a stack:
  - **`server:<config>`** - automatically start a server with the given config (e.g., `server:starter`). This provides one-step testing by auto-starting the server if the port is available, or reusing an existing server if already running.
  - **`server:<config>:<port>`** - same as above but with a custom port (e.g., `server:starter:8322`)
  - a URL which points to a Llama Stack distribution server
  - a distribution name (e.g., `starter`) or a path to a `run.yaml` file
  - a comma-separated list of api=provider pairs, e.g. `inference=ollama,safety=llama-guard,agents=meta-reference`. This is most useful for testing a single API surface.
- `--env`: set environment variables, e.g. --env KEY=value. this is a utility option to set environment variables required by various providers.

Model parameters can be influenced by the following options:
- `--text-model`: comma-separated list of text models.
- `--vision-model`: comma-separated list of vision models.
- `--embedding-model`: comma-separated list of embedding models.
- `--safety-shield`: comma-separated list of safety shields.
- `--judge-model`: comma-separated list of judge models.
- `--embedding-dimension`: output dimensionality of the embedding model to use for testing. Default: 768

Each of these are comma-separated lists and can be used to generate multiple parameter combinations. Note that tests will be skipped
if no model is specified.

### Suites and Setups

- `--suite`: single named suite that narrows which tests are collected.
- Available suites:
  - `base`: collects most tests (excludes responses and post_training)
  - `responses`: collects tests under `tests/integration/responses` (needs strong tool-calling models)
  - `vision`: collects only `tests/integration/inference/test_vision_inference.py`
- `--setup`: global configuration that can be used with any suite. Setups prefill model/env defaults; explicit CLI flags always win.
  - Available setups:
    - `ollama`: Local Ollama provider with lightweight models (sets OLLAMA_URL, uses llama3.2:3b-instruct-fp16)
    - `vllm`: VLLM provider for efficient local inference (sets VLLM_URL, uses Llama-3.2-1B-Instruct)
    - `gpt`: OpenAI GPT models for high-quality responses (uses gpt-4o)
    - `claude`: Anthropic Claude models for high-quality responses (uses claude-3-5-sonnet)

Examples

```bash
# Fast responses run with a strong tool-calling model
pytest -s -v tests/integration --stack-config=server:starter --suite=responses --setup=gpt

# Fast single-file vision run with Ollama defaults
pytest -s -v tests/integration --stack-config=server:starter --suite=vision --setup=ollama

# Base suite with VLLM for performance
pytest -s -v tests/integration --stack-config=server:starter --suite=base --setup=vllm

# Override a default from setup
pytest -s -v tests/integration --stack-config=server:starter \
  --suite=responses --setup=gpt --embedding-model=text-embedding-3-small
```

## Examples

### Testing against a Server

Run all text inference tests by auto-starting a server with the `starter` config:

```bash
OLLAMA_URL=http://localhost:11434 \
  pytest -s -v tests/integration/inference/test_text_inference.py \
   --stack-config=server:starter \
   --text-model=ollama/llama3.2:3b-instruct-fp16 \
   --embedding-model=nomic-embed-text-v1.5
```

Run tests with auto-server startup on a custom port:

```bash
OLLAMA_URL=http://localhost:11434 \
  pytest -s -v tests/integration/inference/ \
   --stack-config=server:starter:8322 \
   --text-model=ollama/llama3.2:3b-instruct-fp16 \
   --embedding-model=nomic-embed-text-v1.5
```

### Testing with Library Client

The library client constructs the Stack "in-process" instead of using a server. This is useful during the iterative development process since you don't need to constantly start and stop servers.


You can do this by simply using `--stack-config=starter` instead of `--stack-config=server:starter`.


### Using ad-hoc distributions

Sometimes, you may want to make up a distribution on the fly. This is useful for testing a single provider or a single API or a small combination of providers. You can do so by specifying a comma-separated list of api=provider pairs to the `--stack-config` option, e.g. `inference=remote::ollama,safety=inline::llama-guard,agents=inline::meta-reference`.

```bash
pytest -s -v tests/integration/inference/ \
   --stack-config=inference=remote::ollama,safety=inline::llama-guard,agents=inline::meta-reference \
   --text-model=$TEXT_MODELS \
   --vision-model=$VISION_MODELS \
   --embedding-model=$EMBEDDING_MODELS
```

Another example: Running Vector IO tests for embedding models:

```bash
pytest -s -v tests/integration/vector_io/ \
   --stack-config=inference=inline::sentence-transformers,vector_io=inline::sqlite-vec \
   --embedding-model=nomic-embed-text-v1.5
```

## Recording Modes

The testing system supports four modes controlled by environment variables:

### REPLAY Mode (Default)
Uses cached responses instead of making API calls:
```bash
pytest tests/integration/
```

### RECORD-IF-MISSING Mode (Recommended for adding new tests)
Records only when no recording exists, otherwise replays. This is the preferred mode for iterative development:
```bash
pytest tests/integration/inference/test_new_feature.py --inference-mode=record-if-missing
```

### RECORD Mode
**Force-records all API interactions**, overwriting existing recordings. Use with caution as this will re-record everything:
```bash
pytest tests/integration/inference/test_new_feature.py --inference-mode=record
```

### LIVE Mode
Tests make real API calls (not recorded):
```bash
pytest tests/integration/ --inference-mode=live
```

By default, the recording directory is `tests/integration/recordings`. You can override this by setting the `LLAMA_STACK_TEST_RECORDING_DIR` environment variable.

## Managing Recordings

### Viewing Recordings
```bash
# See what's recorded
sqlite3 recordings/index.sqlite "SELECT endpoint, model, timestamp FROM recordings;"

# Inspect specific response
cat recordings/responses/abc123.json | jq '.'
```

### Re-recording Tests

#### Remote Re-recording (Recommended)
Use the automated workflow script for easier re-recording:
```bash
./scripts/github/schedule-record-workflow.sh --subdirs "inference,agents"
```
See the [main testing guide](../README.md#remote-re-recording-recommended) for full details.

#### Local Re-recording
```bash
# Re-record specific tests
pytest -s -v --stack-config=server:starter tests/integration/inference/test_modified.py --inference-mode=record
```

Note that when re-recording tests, you must use a Stack pointing to a server (i.e., `server:starter`). This subtlety exists because the set of tests run in server are a superset of the set of tests run in the library client.

## Writing Tests

### Basic Test Pattern
```python
def test_basic_chat_completion(llama_stack_client, text_model_id):
    response = llama_stack_client.chat.completions.create(
        model=text_model_id,
        messages=[{"role": "user", "content": "Hello"}],
    )

    # Test structure, not AI output quality
    assert response.choices[0].message is not None
    assert isinstance(response.choices[0].message.content, str)
    assert len(response.choices[0].message.content) > 0
```

### Provider-Specific Tests
```python
def test_asymmetric_embeddings(llama_stack_client, embedding_model_id):
    if embedding_model_id not in MODELS_SUPPORTING_TASK_TYPE:
        pytest.skip(f"Model {embedding_model_id} doesn't support task types")

    query_response = llama_stack_client.inference.embeddings(
        model_id=embedding_model_id,
        contents=["What is machine learning?"],
        task_type="query",
    )

    assert query_response.embeddings is not None
```

There are two obvious types of tests:

| Type | Location | Purpose |
|------|----------|---------|
| **Unit** | [`tests/unit/`](unit/README.md) | Fast, isolated component testing |
| **Integration** | [`tests/integration/`](integration/README.md) | End-to-end workflows with record-replay |

Both have their place. For unit tests, it is important to create minimal mocks and instead rely more on "fakes". Mocks are too brittle. In either case, tests must be very fast and reliable.

### Record-replay for integration tests

Testing AI applications end-to-end creates some challenges:
- **API costs** accumulate quickly during development and CI
- **Non-deterministic responses** make tests unreliable
- **Multiple providers** require testing the same logic across different APIs

Our solution: **Record real API responses once, replay them for fast, deterministic tests.** This is better than mocking because AI APIs have complex response structures and streaming behavior. Mocks can miss edge cases that real APIs exhibit. A single test can exercise underlying APIs in multiple complex ways making it really hard to mock.

This gives you:
- Cost control - No repeated API calls during development
- Speed - Instant test execution with cached responses
- Reliability - Consistent results regardless of external service state
- Provider coverage - Same tests work across OpenAI, Anthropic, local models, etc.

### Testing Quick Start

You can run the unit tests with:
```bash
uv run --group unit pytest -sv tests/unit/
```

For running integration tests, you must provide a few things:

- A stack config. This is a pointer to a stack. You have a few ways to point to a stack:
  - **`server:<config>`** - automatically start a server with the given config (e.g., `server:starter`). This provides one-step testing by auto-starting the server if the port is available, or reusing an existing server if already running.
  - **`server:<config>:<port>`** - same as above but with a custom port (e.g., `server:starter:8322`)
  - a URL which points to a Llama Stack distribution server
  - a distribution name (e.g., `starter`) or a path to a `run.yaml` file
  - a comma-separated list of api=provider pairs, e.g. `inference=fireworks,safety=llama-guard,agents=meta-reference`. This is most useful for testing a single API surface.

- Whether you are using replay or live mode for inference. This is specified with the LLAMA_STACK_TEST_INFERENCE_MODE environment variable. The default mode currently is "live" -- that is certainly surprising, but we will fix this soon.

- Any API keys you need to use should be set in the environment, or can be passed in with the --env option.

You can run the integration tests in replay mode with:
```bash
# Run all tests with existing recordings
LLAMA_STACK_TEST_INFERENCE_MODE=replay \
  LLAMA_STACK_TEST_RECORDING_DIR=tests/integration/recordings \
  uv run --group test \
  pytest -sv tests/integration/ --stack-config=starter
```

If you don't specify LLAMA_STACK_TEST_INFERENCE_MODE, by default it will be in "live" mode -- that is, it will make real API calls.

```bash
# Test against live APIs
FIREWORKS_API_KEY=your_key pytest -sv tests/integration/inference --stack-config=starter
```

### Re-recording tests

#### Local Re-recording (Manual Setup Required)

If you want to re-record tests locally, you can do so with:

```bash
LLAMA_STACK_TEST_INFERENCE_MODE=record \
  LLAMA_STACK_TEST_RECORDING_DIR=tests/integration/recordings \
  uv run --group test \
  pytest -sv tests/integration/ --stack-config=starter -k "<appropriate test name>"
```

This will record new API responses and overwrite the existing recordings.

```{warning}

You must be careful when re-recording. CI workflows assume a specific setup for running the replay-mode tests. You must re-record the tests in the same way as the CI workflows. This means
- you need Ollama running and serving some specific models.
- you are using the `starter` distribution.
```

#### Remote Re-recording (Recommended)

**For easier re-recording without local setup**, use the automated recording workflow:

```bash
# Record tests for specific test subdirectories
./scripts/github/schedule-record-workflow.sh --test-subdirs "agents,inference"

# Record with vision tests enabled
./scripts/github/schedule-record-workflow.sh --test-subdirs "inference" --run-vision-tests

# Record with specific provider
./scripts/github/schedule-record-workflow.sh --test-subdirs "agents" --test-provider vllm
```

This script:
- 🚀 **Runs in GitHub Actions** - no local Ollama setup required
- 🔍 **Auto-detects your branch** and associated PR
- 🍴 **Works from forks** - handles repository context automatically
- ✅ **Commits recordings back** to your branch

**Prerequisites:**
- GitHub CLI: `brew install gh && gh auth login`
- jq: `brew install jq`
- Your branch pushed to a remote

**Supported providers:** `vllm`, `ollama`


### Next Steps

- [Integration Testing Guide](integration/README.md) - Detailed usage and configuration
- [Unit Testing Guide](unit/README.md) - Fast component testing

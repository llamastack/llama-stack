# Llama Stack Architecture - Comprehensive Overview

## Executive Summary

Llama Stack is a comprehensive framework for building AI applications with Llama models. It provides a **unified API layer** with a **plugin architecture for providers**, allowing developers to seamlessly switch between local and cloud-hosted implementations without changing application code. The system is organized around three main pillars: APIs (abstract interfaces), Providers (concrete implementations), and Distributions (pre-configured bundles).

---

## 1. Core Architecture Philosophy

### Separation of Concerns
- **APIs**: Define abstract interfaces for functionality (e.g., Inference, Safety, VectorIO)
- **Providers**: Implement those interfaces (inline for local, remote for external services)
- **Distributions**: Pre-configure and bundle providers for specific deployment scenarios

### Key Design Patterns
- **Plugin Architecture**: Dynamically load providers based on configuration
- **Dependency Injection**: Providers declare dependencies on other APIs/providers
- **Routing**: Smart routing directs requests to appropriate provider implementations
- **Configuration-Driven**: YAML-based configuration enables flexibility without code changes

---

## 2. Directory Structure (`llama_stack/`)

```
llama_stack/
├── apis/                    # Abstract API definitions (27 APIs total)
│   ├── inference/          # LLM inference interface
│   ├── agents/             # Agent orchestration
│   ├── safety/             # Content filtering & safety
│   ├── vector_io/          # Vector database operations
│   ├── tools/              # Tool/function calling runtime
│   ├── scoring/            # Response scoring
│   ├── eval/               # Evaluation framework
│   ├── post_training/      # Fine-tuning & training
│   ├── datasetio/          # Dataset loading/management
│   ├── conversations/      # Conversation management
│   ├── common/             # Shared datatypes (SamplingParams, etc.)
│   └── [22 more...]        # Models, Shields, Benchmarks, etc.
│
├── providers/              # Provider implementations (inline & remote)
│   ├── inline/             # In-process implementations
│   │   ├── inference/      # Meta Reference, Sentence Transformers
│   │   ├── agents/         # Agent orchestration implementations
│   │   ├── safety/         # Llama Guard, Code Scanner
│   │   ├── vector_io/      # FAISS, SQLite-vec, Milvus
│   │   ├── post_training/  # TorchTune
│   │   ├── eval/           # Evaluation implementations
│   │   ├── tool_runtime/   # RAG runtime, MCP protocol
│   │   └── [more...]
│   │
│   ├── remote/             # External service adapters
│   │   ├── inference/      # OpenAI, Anthropic, Groq, Ollama, vLLM, TGI, etc.
│   │   ├── vector_io/      # ChromaDB, Qdrant, Weaviate, Postgres
│   │   ├── safety/         # Bedrock, SambaNova, Nvidia
│   │   ├── agents/         # Sample implementations
│   │   ├── tool_runtime/   # Brave Search, Tavily, Wolfram Alpha
│   │   └── [more...]
│   │
│   ├── registry/           # Provider discovery/registration (inference.py, agents.py, etc.)
│   │   └── [One file per API with all providers for that API]
│   │
│   ├── utils/              # Shared provider utilities
│   │   ├── inference/      # Embedding mixin, OpenAI compat
│   │   ├── kvstore/        # Key-value store abstractions
│   │   ├── sqlstore/       # SQL storage abstractions
│   │   ├── telemetry/      # Tracing, metrics
│   │   └── [more...]
│   │
│   └── datatypes.py        # ProviderSpec, InlineProviderSpec, RemoteProviderSpec
│
├── core/                   # Core runtime & orchestration
│   ├── stack.py            # Main LlamaStack class (implements all APIs)
│   ├── datatypes.py        # Config models (StackRunConfig, Provider, etc.)
│   ├── resolver.py         # Provider resolution & dependency injection
│   ├── library_client.py   # In-process client for library usage
│   ├── build.py            # Distribution building
│   ├── configure.py        # Configuration handling
│   ├── distribution.py     # Distribution management
│   ├── routers/            # Auto-routed API implementations (infer route based on routing key)
│   ├── routing_tables/     # Manual routing tables (e.g., Models, Shields, VectorStores)
│   ├── server/             # FastAPI HTTP server setup
│   ├── storage/            # Backend storage abstractions (KVStore, SqlStore)
│   ├── utils/              # Config resolution, dynamic imports
│   └── conversations/      # Conversation service implementation
│
├── cli/                    # Command-line interface
│   ├── llama.py            # Main entry point
│   └── stack/              # Stack management commands
│       ├── run.py          # Start a distribution
│       ├── list_apis.py    # List available APIs
│       ├── list_providers.py # List providers
│       ├── list_deps.py    # List dependencies
│       └── [more...]
│
├── distributions/          # Pre-configured distribution templates
│   ├── starter/            # CPU-friendly multi-provider starter
│   ├── starter-gpu/        # GPU-optimized starter
│   ├── meta-reference-gpu/ # Full-featured Meta reference
│   ├── postgres-demo/      # PostgreSQL-based demo
│   ├── template.py         # Distribution template base class
│   └── [more...]
│
├── models/                 # Llama model implementations
│   └── llama/
│       ├── llama3/         # Llama 3 implementation
│       ├── llama4/         # Llama 4 implementation
│       ├── sku_list.py     # Model registry (maps model IDs to implementations)
│       ├── checkpoint.py   # Model checkpoint handling
│       ├── datatypes.py    # ToolDefinition, StopReason, etc.
│       └── [more...]
│
├── testing/                # Testing utilities
│   └── api_recorder.py     # Record/replay infrastructure for integration tests
│
└── ui/                     # Web UI (Streamlit-based)
    ├── app/
    ├── components/
    ├── pages/
    └── [React/TypeScript frontend]
```

---

## 3. API Layer (27 APIs)

### What is an API?
Each API is an abstract **protocol** (Python Protocol class) that defines an interface. APIs are located in `llama_stack/apis/` with a structure like:

```
apis/inference/
├── __init__.py          # Exports the Inference protocol
├── inference.py         # Full API definition (300+ lines)
└── event_logger.py      # Supporting types
```

### Key APIs

#### Core Inference API
- **Path**: `llama_stack/apis/inference/inference.py`
- **Methods**: `post_chat_completion()`, `post_completion()`, `post_embedding()`, `get_models()`
- **Types**: `SamplingParams`, `SamplingStrategy` (greedy/top-p/top-k), `OpenAIChatCompletion`
- **Providers**: 30+ (OpenAI, Claude, Ollama, vLLM, TGI, Fireworks, etc.)

#### Agents API
- **Path**: `llama_stack/apis/agents/agents.py`
- **Methods**: `create_agent()`, `update_agent()`, `create_session()`, `agentic_loop_turn()`
- **Features**: Multi-turn conversations, tool calling, streaming
- **Providers**: Meta Reference (inline), Fireworks, Together

#### Safety API
- **Path**: `llama_stack/apis/safety/safety.py`
- **Methods**: `run_shields()` - filter content before/after inference
- **Providers**: Llama Guard (inline), AWS Bedrock, SambaNova, Nvidia

#### Vector IO API
- **Path**: `llama_stack/apis/vector_io/vector_io.py`
- **Methods**: `insert()`, `query()`, `delete()` - vector database operations
- **Providers**: FAISS, SQLite-vec, Milvus (inline), ChromaDB, Qdrant, Weaviate, PG Vector (remote)

#### Tools / Tool Runtime API
- **Path**: `llama_stack/apis/tools/tool_runtime.py`
- **Methods**: `execute_tool()` - execute functions during agent loops
- **Providers**: RAG runtime (inline), Brave Search, Tavily, Wolfram Alpha, Model Context Protocol

#### Other Major APIs
- **Post Training**: Fine-tuning & model training (HuggingFace, TorchTune, Nvidia)
- **Eval**: Evaluation frameworks (Meta Reference with autoevals)
- **Scoring**: Response scoring (Basic, LLM-as-Judge, Braintrust)
- **Datasets**: Dataset management
- **DatasetIO**: Dataset loading from HuggingFace, Nvidia, local files
- **Conversations**: Multi-turn conversation state management
- **Vector Stores**: Vector store metadata & configuration
- **Shields**: Shield (safety filter) registry
- **Models**: Model registry management
- **Batches**: Batch processing
- **Prompts**: Prompt templates & management
- **Telemetry**: Tracing & metrics collection
- **Inspect**: Introspection & debugging

---

## 4. Provider System

### Provider Types

#### 1. **Inline Providers** (`InlineProviderSpec`)
- Run in-process (same Python process as server)
- High performance, low latency
- No network overhead
- Heavier resource requirements
- Examples: Meta Reference (inference), Llama Guard (safety), FAISS (vector IO)

**Structure**:
```python
InlineProviderSpec(
    api=Api.inference,
    provider_type="inline::meta-reference",
    module="llama_stack.providers.inline.inference.meta_reference",
    config_class="...MetaReferenceInferenceConfig",
    pip_packages=[...],
    container_image="..."  # Optional for containerization
)
```

#### 2. **Remote Providers** (`RemoteProviderSpec`)
- Connect to external services via HTTP/API
- Lower resource requirements
- Network latency
- Cloud-based (OpenAI, Anthropic, Groq) or self-hosted (Ollama, vLLM, Qdrant)
- Examples: OpenAI, Anthropic, Groq, Ollama, Qdrant, ChromaDB

**Structure**:
```python
RemoteProviderSpec(
    api=Api.inference,
    adapter_type="openai",
    provider_type="remote::openai",
    module="llama_stack.providers.remote.inference.openai",
    config_class="...OpenAIInferenceConfig",
    pip_packages=[...]
)
```

### Provider Registration

Providers are registered in **registry files** (`llama_stack/providers/registry/`):
- `inference.py` - All inference providers (30+)
- `agents.py` - All agent providers
- `safety.py` - All safety providers
- `vector_io.py` - All vector IO providers
- `tool_runtime.py` - All tool runtime providers
- [etc.]

Each registry file has an `available_providers()` function returning a list of `ProviderSpec`.

### Provider Config

Each provider has a config class (e.g., `MetaReferenceInferenceConfig`):
```python
class MetaReferenceInferenceConfig(BaseModel):
    max_batch_size: int = 1
    enable_pydantic_sampling: bool = True
    # sample_run_config() - provides default values for testing
    # pip_packages() - lists dependencies
```

### Provider Implementation

Inline providers look like:
```python
class MetaReferenceInferenceImpl(InferenceProvider):
    async def post_chat_completion(
        self,
        model: str,
        request: OpenAIChatCompletionRequestWithExtraBody,
    ) -> AsyncIterator[OpenAIChatCompletionChunk]:
        # Load model, run inference, yield streaming results
        ...
```

Remote providers implement HTTP adapters:
```python
class OllamaInferenceImpl(InferenceProvider):
    async def post_chat_completion(...):
        # Make HTTP requests to Ollama server
        ...
```

---

## 5. Core Runtime & Resolution

### Stack Resolution Process

**File**: `llama_stack/core/resolver.py`

1. **Load Configuration** → Parse `run.yaml` with enabled providers
2. **Resolve Dependencies** → Build dependency graph (e.g., agents may depend on inference)
3. **Instantiate Providers** → Create provider instances with configs
4. **Create Router/Routed Impls** → Set up request routing
5. **Register Resources** → Register models, shields, datasets, etc.

### The LlamaStack Class

**File**: `llama_stack/core/stack.py`

```python
class LlamaStack(
    Providers,      # Meta API for provider management
    Inference,      # LLM inference
    Agents,         # Agent orchestration
    Safety,         # Content safety
    VectorIO,       # Vector operations
    Tools,          # Tool runtime
    Eval,           # Evaluation
    # ... 15 more APIs ...
):
    pass
```

This class **inherits from all APIs**, making a single `LlamaStack` instance support all functionality.

### Two Client Modes

#### 1. **Library Client** (In-Process)
```python
from llama_stack import AsyncLlamaStackAsLibraryClient

client = await AsyncLlamaStackAsLibraryClient.create(run_config)
response = await client.inference.post_chat_completion(...)
```
**File**: `llama_stack/core/library_client.py`

#### 2. **Server Client** (HTTP)
```python
from llama_stack_client import AsyncLlamaStackClient

client = AsyncLlamaStackClient(base_url="http://localhost:8321")
response = await client.inference.post_chat_completion(...)
```
Uses the separate `llama-stack-client` package.

---

## 6. Request Routing

### Two Routing Strategies

#### 1. **Auto-Routed APIs** (e.g., Inference, Safety, VectorIO)
- Routing key = provider instance
- Router automatically selects provider based on resource ID
- **Implementation**: `AutoRoutedProviderSpec` → `routers/` directory

```python
# inference.post_chat_completion(model_id="meta-llama/Llama-2-7b")
# Router selects provider based on which provider has that model
```

**Routed APIs**:
- Inference, Safety, VectorIO, DatasetIO, Scoring, Eval, ToolRuntime

#### 2. **Routing Table APIs** (e.g., Models, Shields, VectorStores)
- Registry APIs that list/register resources
- **Implementation**: `RoutingTableProviderSpec` → `routing_tables/` directory

```python
# models.list_models() → merged list from all providers
# models.register_model(...) → router selects provider
```

**Registry APIs**:
- Models, Shields, VectorStores, Datasets, ScoringFunctions, Benchmarks, ToolGroups

---

## 7. Distributions

### What is a Distribution?

A **Distribution** is a pre-configured, verified bundle of providers for a specific deployment scenario.

**File**: `llama_stack/distributions/template.py` (base) → specific distros in subdirectories

### Example: Starter Distribution

**File**: `llama_stack/distributions/starter/starter.py`

```python
def get_distribution_template(name: str = "starter"):
    providers = {
        "inference": [
            remote::ollama,
            remote::vllm,
            remote::openai,
            # ... others ...
        ],
        "vector_io": [
            inline::faiss,
            inline::sqlite-vec,
            remote::qdrant,
            # ... others ...
        ],
        "safety": [
            inline::llama-guard,
            inline::code-scanner,
        ],
        # ... other APIs ...
    }
    return DistributionTemplate(
        name="starter",
        providers=providers,
        run_configs={
            "run.yaml": RunConfigSettings(...)
        }
    )
```

### Built-in Distributions

1. **starter**: CPU-only, multi-provider (Ollama, OpenAI, etc.)
2. **starter-gpu**: GPU-optimized version
3. **meta-reference-gpu**: Full Meta reference implementation
4. **postgres-demo**: PostgreSQL-backed version
5. **watsonx**: IBM Watson X integration
6. **nvidia**: NVIDIA-specific optimizations
7. **open-benchmark**: For benchmarking

### Distribution Lifecycle

```
llama stack run starter
  ↓
Resolve starter distribution template
  ↓
Merge with run.yaml config & environment variables
  ↓
Build/install dependencies (if needed)
  ↓
Start HTTP server (Uvicorn)
  ↓
Initialize all providers
  ↓
Register resources (models, shields, etc.)
  ↓
Ready for requests
```

---

## 8. CLI Architecture

**File**: `llama_stack/cli/`

### Entry Point

```bash
$ llama [subcommand] [args]
```

Maps to **pyproject.toml**:
```toml
[project.scripts]
llama = "llama_stack.cli.llama:main"
```

### Subcommands

```
llama stack [command]
  ├── run [distro|config] [--port PORT]     # Start a distribution
  ├── list-deps [distro]                    # Show dependencies to install
  ├── list-apis                             # Show all APIs
  ├── list-providers                        # Show all providers
  └── list [NAME]                           # Show distributions
```

**Architecture**:
- `llama.py` - Main parser with subcommands
- `stack/stack.py` - Stack subcommand router
- `stack/run.py` - Implementation of `llama stack run`
- `stack/list_deps.py` - Dependency resolution & display

---

## 9. Testing Architecture

**Location**: `tests/` directory

### Test Types

#### 1. **Unit Tests** (`tests/unit/`)
- Fast, isolated component testing
- Mock external dependencies
- **Run with**: `uv run --group unit pytest tests/unit/`
- **Examples**:
  - `core/test_stack_validation.py` - Config validation
  - `distribution/test_distribution.py` - Distribution loading
  - `core/routers/test_vector_io.py` - Routing logic

#### 2. **Integration Tests** (`tests/integration/`)
- End-to-end workflows
- **Record-Replay pattern**: Record real API responses once, replay for fast/cheap testing
- **Run with**: `uv run --group test pytest tests/integration/ --stack-config=starter`
- **Structure**:
  ```
  tests/integration/
  ├── agents/
  │   ├── test_agents.py
  │   ├── test_persistence.py
  │   └── cassettes/  # Recorded API responses (YAML)
  ├── inference/
  ├── safety/
  ├── vector_io/
  └── [more...]
  ```

### Record-Replay System

**File**: `llama_stack/testing/api_recorder.py`

**Benefits**:
- **Cost Control**: Record real API calls once, replay thousands of times
- **Speed**: Cached responses = instant test execution
- **Reliability**: Deterministic results (no API variability)
- **Provider Coverage**: Same test works with OpenAI, Anthropic, Ollama, etc.

**How it works**:
1. First run (with `LLAMA_STACK_TEST_INFERENCE_MODE=record`): Real API calls saved to YAML
2. Subsequent runs: Load YAML and return matching responses
3. CI automatically re-records when needed

### Test Organization

- **Common utilities**: `tests/common/`
- **External provider tests**: `tests/external/` (test external APIs)
- **Container tests**: `tests/containers/` (test Docker integration)
- **Conftest**: pytest fixtures in each directory

---

## 10. Key Design Patterns

### Pattern 1: Protocol-Based Abstraction
```python
# API definition (protocol)
class Inference(Protocol):
    async def post_chat_completion(...) -> AsyncIterator[...]: ...

# Provider implementation
class InferenceProvider:
    async def post_chat_completion(...): ...
```

### Pattern 2: Dependency Injection
```python
class AgentProvider:
    def __init__(self, inference: InferenceProvider, safety: SafetyProvider):
        self.inference = inference
        self.safety = safety
```

### Pattern 3: Configuration-Driven Instantiation
```yaml
# run.yaml
agents:
  - provider_id: meta-reference
    provider_type: inline::meta-reference
    config:
      max_depth: 5
```

### Pattern 4: Routing by Resource
```python
# Request: inference.post_chat_completion(model="llama-2-7b")
# Router finds which provider has "llama-2-7b" and routes there
```

### Pattern 5: Registry Pattern for Resources
```python
# Register at startup
await models.register_model(Model(
    identifier="llama-2-7b",
    provider_id="inference::meta-reference",
    ...
))

# Later, query or filter
models_list = await models.list_models()
```

---

## 11. Configuration Management

### Config Files

#### 1. **run.yaml** - Runtime Configuration
Location: `~/.llama/distributions/{name}/run.yaml`

```yaml
version: 2
providers:
  inference:
    - provider_id: ollama
      provider_type: remote::ollama
      config:
        host: localhost
        port: 11434
  safety:
    - provider_id: llama-guard
      provider_type: inline::llama-guard
      config: {}
default_models:
  - identifier: llama-2-7b
    provider_id: ollama
vector_stores_config:
  default_provider_id: faiss
```

#### 2. **build.yaml** - Build Configuration
Specifies which providers to install.

#### 3. Environment Variables
Override config values at runtime:
```bash
INFERENCE_MODEL=llama-2-70b SAFETY_MODEL=llama-guard llama stack run starter
```

### Config Resolution

**File**: `llama_stack/core/utils/config_resolution.py`

Order of precedence:
1. Environment variables (highest)
2. Runtime config (run.yaml)
3. Distribution template defaults (lowest)

---

## 12. Extension Points for Developers

### Adding a Custom Provider

1. **Create provider module**:
   ```python
   llama_stack/providers/remote/inference/my_provider/
   ├── __init__.py
   ├── config.py          # MyProviderConfig
   └── my_provider.py     # MyProviderImpl(InferenceProvider)
   ```

2. **Register in registry**:
   ```python
   # llama_stack/providers/registry/inference.py
   RemoteProviderSpec(
       api=Api.inference,
       adapter_type="my_provider",
       provider_type="remote::my_provider",
       config_class="...MyProviderConfig",
       module="llama_stack.providers.remote.inference.my_provider",
   )
   ```

3. **Use in distribution**:
   ```yaml
   providers:
     inference:
       - provider_id: my_provider
         provider_type: remote::my_provider
         config: {...}
   ```

### Adding a Custom API

1. Define protocol in `llama_stack/apis/my_api/my_api.py`
2. Implement providers
3. Register in resolver and distributions
4. Add CLI support if needed

---

## 13. Storage & Persistence

### Storage Backends

**File**: `llama_stack/core/storage/datatypes.py`

#### KV Store (Key-Value)
- Store metadata: models, shields, vector stores
- Backends: SQLite (inline), Redis, Postgres

#### SQL Store
- Store structured data: conversations, datasets
- Backends: SQLite (inline), Postgres

#### Inference Store
- Cache inference results for recording/replay
- Used in testing

### Storage Configuration

```yaml
storage:
  type: sqlite
  config:
    dir: ~/.llama/distributions/starter
```

---

## 14. Telemetry & Tracing

### Tracing System

**File**: `llama_stack/providers/utils/telemetry/`

- Automatic request tracing with OpenTelemetry
- Trace context propagation across async calls
- Integration with OpenTelemetry collectors

### Telemetry API

Providers can implement the Telemetry API to collect metrics:
- Token usage
- Latency
- Error rates
- Custom metrics

---

## 15. Model System

### Model Registry

**File**: `llama_stack/models/llama/sku_list.py`

```python
resolve_model("meta-llama/Llama-2-7b") 
  → Llama2Model(...)
```

Maps model IDs to their:
- Architecture
- Tokenizer
- Quantization options
- Required resources

### Supported Models

- **Llama 3** - Full architecture support
- **Llama 3.1** - Extended context
- **Llama 3.2** - Multimodal support
- **Llama 4** - Latest generation
- **Custom models** - Via provider registration

### Model Quantization

- int8, int4
- GPTQ
- Hadamard transform
- Custom quantizers

---

## 16. Key Files to Understand

### For Understanding Core Concepts
1. `llama_stack/core/datatypes.py` - Configuration data types
2. `llama_stack/providers/datatypes.py` - Provider specs
3. `llama_stack/apis/inference/inference.py` - Example API

### For Understanding Runtime
1. `llama_stack/core/stack.py` - Main runtime class
2. `llama_stack/core/resolver.py` - Dependency resolution
3. `llama_stack/core/library_client.py` - In-process client

### For Understanding Providers
1. `llama_stack/providers/registry/inference.py` - Inference provider registry
2. `llama_stack/providers/inline/inference/meta_reference/inference.py` - Example inline
3. `llama_stack/providers/remote/inference/openai/openai.py` - Example remote

### For Understanding Distributions
1. `llama_stack/distributions/template.py` - Distribution template
2. `llama_stack/distributions/starter/starter.py` - Starter distro
3. `llama_stack/cli/stack/run.py` - Distribution startup

---

## 17. Development Workflow

### Running Locally

```bash
# Install dependencies
uv sync --all-groups

# Run a distribution (auto-starts server)
llama stack run starter

# In another terminal, interact with it
curl http://localhost:8321/health
```

### Testing

```bash
# Unit tests (fast, no external dependencies)
uv run --group unit pytest tests/unit/

# Integration tests (with record-replay)
uv run --group test pytest tests/integration/ --stack-config=starter

# Re-record integration tests (record real API calls)
LLAMA_STACK_TEST_INFERENCE_MODE=record \
  uv run --group test pytest tests/integration/ --stack-config=starter
```

### Building Distributions

```bash
# Build Starter distribution
llama stack build starter --name my-starter

# Run it
llama stack run my-starter
```

---

## 18. Notable Implementation Details

### Async-First Architecture
- All I/O is async (using `asyncio`)
- Streaming responses with `AsyncIterator`
- FastAPI for HTTP server (built on Starlette)

### Streaming Support
- Inference responses stream tokens
- Agents stream turn-by-turn updates
- Proper async context preservation

### Error Handling
- Structured errors with detailed messages
- Graceful degradation when dependencies unavailable
- Provider health checks

### Extensibility
- External providers via module import
- Custom APIs via ExternalApiSpec
- Plugin discovery via provider registry

---

## 19. Typical Request Flow

```
User Request (e.g., chat completion)
  ↓
CLI or SDK Client
  ↓
HTTP Request → FastAPI Server (port 8321)
  ↓
Route Handler (e.g., /inference/chat-completion)
  ↓
Router (Auto-Routed API)
  → Determine which provider has the model
  ↓
Provider Implementation (e.g., OpenAI, Ollama, Meta Reference)
  ↓
External Service or Local Execution
  ↓
Response (streaming or complete)
  ↓
Send back to Client
```

---

## 20. Key Takeaways

1. **Unified APIs**: Single abstraction for 27+ AI capabilities
2. **Pluggable Providers**: 50+ implementations (inline & remote)
3. **Configuration-Driven**: Switch providers via YAML, not code
4. **Distributions**: Pre-verified bundles for common scenarios
5. **Record-Replay Testing**: Cost-effective integration tests
6. **Two Client Modes**: Library (in-process) or HTTP (distributed)
7. **Smart Routing**: Automatic request routing to appropriate providers
8. **Async-First**: Native streaming and concurrent request handling
9. **Extensible**: Custom APIs and providers easily added
10. **Production-Ready**: Health checks, telemetry, access control, storage

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Applications                      │
│               (CLI, SDK, Web UI, Custom Apps)               │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┴────────────┐
         │                        │
    ┌────▼────────┐      ┌───────▼──────┐
    │   Library   │      │  HTTP Server │
    │   Client    │      │  (FastAPI)   │
    └────┬────────┘      └───────┬──────┘
         │                       │
         └───────────┬───────────┘
                     │
          ┌──────────▼──────────┐
          │   LlamaStack Class  │
          │  (implements all    │
          │   27 APIs)          │
          └──────────┬──────────┘
                     │
      ┌──────────────┼──────────────┐
      │              │              │
      │    Router    │   Routing    │  Resource
      │  (Auto-      │   Tables     │  Registries
      │   routed     │  (Models,    │  (Models,
      │   APIs)      │   Shields)   │   Shields,
      │              │              │   etc.)
      └──────────────┼──────────────┘
                     │
        ┌────────────┴──────────────┐
        │                           │
   ┌────▼──────────┐    ┌──────────▼─────┐
   │ Inline        │    │ Remote          │
   │ Providers     │    │ Providers       │
   │               │    │                 │
   │ • Meta Ref    │    │ • OpenAI        │
   │ • FAISS       │    │ • Ollama        │
   │ • Llama Guard │    │ • Qdrant        │
   │ • etc.        │    │ • etc.          │
   │               │    │                 │
   └───────────────┘    └─────────────────┘
        │                       │
        │                       │
   Local Execution         External Services
   (GPUs/CPUs)            (APIs/Servers)
```


# Llama Stack - Quick Reference Guide

## Key Concepts at a Glance

### The Three Pillars
1. **APIs** (`llama_stack/apis/`) - Abstract interfaces (27 total)
2. **Providers** (`llama_stack/providers/`) - Implementations (50+ total)
3. **Distributions** (`llama_stack/distributions/`) - Pre-configured bundles

### Directory Map for Quick Navigation

| Component | Location | Purpose |
|-----------|----------|---------|
| Inference API | `apis/inference/inference.py` | LLM chat, completion, embeddings |
| Agents API | `apis/agents/agents.py` | Multi-turn agent orchestration |
| Safety API | `apis/safety/safety.py` | Content filtering |
| Vector IO API | `apis/vector_io/vector_io.py` | Vector database operations |
| Core Stack | `core/stack.py` | Main orchestrator (implements all APIs) |
| Provider Resolver | `core/resolver.py` | Dependency injection & instantiation |
| Inline Inference | `providers/inline/inference/` | Local model execution |
| Remote Inference | `providers/remote/inference/` | API providers (OpenAI, Ollama, etc.) |
| CLI Entry Point | `cli/llama.py` | Command-line interface |
| Starter Distribution | `distributions/starter/` | Basic multi-provider setup |

## Common Tasks

### Understanding an API
1. Read the API definition: `llama_stack/apis/[api_name]/[api_name].py`
2. Check common types: `llama_stack/apis/common/`
3. Look at providers: `llama_stack/providers/registry/[api_name].py`
4. Examine an implementation: `llama_stack/providers/inline/[api_name]/[provider]/`

### Adding a Provider
1. Create module: `llama_stack/providers/remote/[api]/[provider_name]/`
2. Implement class extending the API protocol
3. Register in: `llama_stack/providers/registry/[api].py`
4. Add to distribution: `llama_stack/distributions/[distro]/[distro].py`

### Debugging a Request
1. Check routing: `llama_stack/core/routers/` or `routing_tables/`
2. Find provider: `llama_stack/providers/registry/[api].py`
3. Read implementation: `llama_stack/providers/[inline|remote]/[api]/[provider]/`
4. Check config: Look for `Config` class in provider module

### Running Tests
```bash
# Unit tests (fast)
uv run --group unit pytest tests/unit/

# Integration tests (with replay)
uv run --group test pytest tests/integration/ --stack-config=starter

# Re-record tests
LLAMA_STACK_TEST_INFERENCE_MODE=record uv run --group test pytest tests/integration/
```

## Core Classes to Know

### ProviderSpec Hierarchy
```
ProviderSpec (base)
├── InlineProviderSpec (in-process)
└── RemoteProviderSpec (external services)
```

### Key Runtime Classes
- **LlamaStack** (`core/stack.py`) - Main class implementing all APIs
- **StackRunConfig** (`core/datatypes.py`) - Configuration for a stack
- **ProviderRegistry** (`core/resolver.py`) - Maps APIs to providers

### Key Data Classes
- **Provider** - Concrete provider instance with config
- **Model** - Registered model (from a provider)
- **OpenAIChatCompletion** - Response format (from Inference API)

## Configuration Files

### run.yaml Structure
```yaml
version: 2
providers:
  [api_name]:
    - provider_id: unique_name
      provider_type: inline::name or remote::name
      config: {}  # Provider-specific config
default_models:
  - identifier: model_id
    provider_id: inference_provider_id
vector_stores_config:
  default_provider_id: faiss_or_other
```

### Environment Variables
Override any config value:
```bash
INFERENCE_MODEL=llama-2-7b llama stack run starter
```

## Common File Patterns

### Inline Provider Structure
```
llama_stack/providers/inline/[api]/[provider]/
├── __init__.py          # Exports adapter class
├── config.py            # ConfigClass
├── [provider].py        # AdapterImpl(ProtocolClass)
└── [utils].py           # Helper modules
```

### Remote Provider Structure  
```
llama_stack/providers/remote/[api]/[provider]/
├── __init__.py          # Exports adapter class
├── config.py            # ConfigClass
└── [provider].py        # AdapterImpl with HTTP calls
```

### API Structure
```
llama_stack/apis/[api]/
├── __init__.py          # Exports main protocol
├── [api].py             # Main protocol definition
└── [supporting].py      # Types and supporting classes
```

## Key Design Patterns

### Pattern 1: Auto-Routed APIs
Provider selected automatically based on resource ID
```python
# Router finds which provider has this model
await inference.post_chat_completion(model="llama-2-7b")
```

### Pattern 2: Routing Tables
Registry APIs that list/register resources
```python
# Returns merged list from all providers
await models.list_models()

# Router selects provider internally
await models.register_model(model)
```

### Pattern 3: Dependency Injection
Providers depend on other APIs
```python
class AgentProvider:
    def __init__(self, inference: InferenceProvider, ...):
        self.inference = inference
```

## Important Numbers

- **27 APIs** total in Llama Stack
- **30+ Inference Providers** (OpenAI, Anthropic, Groq, local, etc.)
- **10+ Vector IO Providers** (FAISS, Qdrant, ChromaDB, etc.)
- **5+ Safety Providers** (Llama Guard, Bedrock, etc.)
- **7 Built-in Distributions** (starter, starter-gpu, meta-reference-gpu, etc.)

## Quick Commands

```bash
# List all APIs
llama stack list-apis

# List all providers
llama stack list-providers [api_name]

# List distributions
llama stack list

# Show dependencies for a distribution
llama stack list-deps starter

# Start a distribution on custom port
llama stack run starter --port 8322

# Interact with running server
curl http://localhost:8321/health
```

## File Size Reference (to judge complexity)

| File | Size | Complexity |
|------|------|-----------|
| inference.py (API) | 46KB | High (30+ parameters) |
| stack.py (core) | 21KB | High (orchestration) |
| resolver.py (core) | 19KB | High (dependency resolution) |
| library_client.py (core) | 20KB | Medium (client implementation) |
| template.py (distributions) | 18KB | Medium (config generation) |

## Testing Quick Reference

### Record-Replay Testing
1. **Record**: `LLAMA_STACK_TEST_INFERENCE_MODE=record pytest ...`
2. **Replay**: `pytest ...` (default, no network calls)
3. **Location**: `tests/integration/[api]/cassettes/`
4. **Format**: YAML files with request/response pairs

### Test Structure
- Unit tests: No external dependencies
- Integration tests: Use actual providers (record-replay)
- Common fixtures: `tests/unit/conftest.py`, `tests/integration/conftest.py`

## Common Debugging Tips

1. **Provider not loading?** → Check `llama_stack/providers/registry/[api].py`
2. **Config validation error?** → Check provider's `Config` class
3. **Import error?** → Verify `pip_packages` in ProviderSpec
4. **Routing not working?** → Check `llama_stack/core/routers/` or `routing_tables/`
5. **Test failing?** → Check cassettes in `tests/integration/[api]/cassettes/`

## Most Important Files for Beginners

1. `pyproject.toml` - Project metadata & entry points
2. `llama_stack/core/stack.py` - Understand the main class
3. `llama_stack/core/resolver.py` - Understand how providers are loaded
4. `llama_stack/apis/inference/inference.py` - Understand an API
5. `llama_stack/providers/registry/inference.py` - See all inference providers
6. `llama_stack/distributions/starter/starter.py` - See how distributions work


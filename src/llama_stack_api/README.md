# llama-stack-api

API and Provider specifications for Llama Stack - a lightweight package with protocol definitions and provider specs.

## Overview

`llama-stack-api` is a minimal dependency package that contains:

- **API Protocol Definitions**: Type-safe protocol definitions for all Llama Stack APIs (inference, agents, safety, etc.)
- **Provider Specifications**: Provider spec definitions for building custom providers
- **Data Types**: Shared data types and models used across the Llama Stack ecosystem
- **Type Utilities**: Strong typing utilities and schema validation

## What This Package Does NOT Include

- Server implementation (see `llama-stack` package)
- Provider implementations (see `llama-stack` package)
- CLI tools (see `llama-stack` package)
- Runtime orchestration (see `llama-stack` package)

## Use Cases

This package is designed for:

1. **Third-party Provider Developers**: Build custom providers without depending on the full Llama Stack server
2. **Client Library Authors**: Use type definitions without server dependencies
3. **Documentation Generation**: Generate API docs from protocol definitions
4. **Type Checking**: Validate implementations against the official specs

## Installation

```bash
pip install llama-stack-api
```

Or with uv:

```bash
uv pip install llama-stack-api
```

## Dependencies

Minimal dependencies:
- `pydantic>=2.11.9` - For data validation and serialization
- `jsonschema` - For JSON schema utilities

## Versioning

This package follows semantic versioning independently from the main `llama-stack` package:

- **Patch versions** (0.1.x): Documentation, internal improvements
- **Minor versions** (0.x.0): New APIs, backward-compatible changes
- **Major versions** (x.0.0): Breaking changes to existing APIs

Current version: **0.4.0.dev0**

## Usage Example

```python
from llama_stack_api.inference import Inference, ChatCompletionRequest
from llama_stack_api.providers.datatypes import ProviderSpec, InlineProviderSpec
from llama_stack_api.datatypes import Api


# Use protocol definitions for type checking
class MyInferenceProvider(Inference):
    async def chat_completion(self, request: ChatCompletionRequest):
        # Your implementation
        pass


# Define provider specifications
my_provider_spec = InlineProviderSpec(
    api=Api.inference,
    provider_type="inline::my-provider",
    pip_packages=["my-dependencies"],
    module="my_package.providers.inference",
    config_class="my_package.providers.inference.MyConfig",
)
```

## Relationship to llama-stack

The main `llama-stack` package depends on `llama-stack-api` and provides:
- Full server implementation
- Built-in provider implementations
- CLI tools for running and managing stacks
- Runtime provider resolution and orchestration

## Contributing

See the main [Llama Stack repository](https://github.com/llamastack/llama-stack) for contribution guidelines.

## License

MIT License - see LICENSE file for details.

## Links

- [Main Llama Stack Repository](https://github.com/llamastack/llama-stack)
- [Documentation](https://llamastack.ai/)
- [Client Library](https://pypi.org/project/llama-stack-client/)

# Adding a New API Provider

This guide will walk you through the process of adding a new API provider to Llama Stack.


- Begin by reviewing the [core concepts](../concepts/index.md) of Llama Stack and choose the API your provider belongs to (Inference, Safety, VectorIO, etc.)
- Determine the provider type ({repopath}`Remote::llama_stack/providers/remote` or {repopath}`Inline::llama_stack/providers/inline`). Remote providers make requests to external services, while inline providers execute implementation locally.
- Add your provider to the appropriate {repopath}`Registry::llama_stack/providers/registry/`. Specify pip dependencies necessary.
- Update any distribution {repopath}`Templates::llama_stack/distributions/` `build.yaml` and `run.yaml` files if they should include your provider by default. Run {repopath}`./scripts/distro_codegen.py` if necessary. Note that `distro_codegen.py` will fail if the new provider causes any distribution template to attempt to import provider-specific dependencies. This usually means the distribution's `get_distribution_template()` code path should only import any necessary Config or model alias definitions from each provider and not the provider's actual implementation.


Here are some example PRs to help you get started:
   - [Grok Inference Implementation](https://github.com/meta-llama/llama-stack/pull/609)
   - [Nvidia Inference Implementation](https://github.com/meta-llama/llama-stack/pull/355)
   - [Model context protocol Tool Runtime](https://github.com/meta-llama/llama-stack/pull/665)

## Guidelines for creating Internal or External Providers

|**Type** |Internal (In-tree) |External (out-of-tree)
|---------|-------------------|---------------------|
|**Description** |A provider that is directly in the Llama Stack code|A provider that is outside of the Llama stack core codebase but is still accessible and usable by Llama Stack.
|**Benefits** |Ability to interact with the provider with minimal additional configurations or installations| Contributors do not have to add directly to the code to create providers accessible on Llama Stack. Keep provider-specific code separate from the core Llama Stack code.

## Inference Provider Patterns

When implementing Inference providers for OpenAI-compatible APIs, Llama Stack provides several mixin classes to simplify development and ensure consistent behavior across providers.

### OpenAIMixin

The `OpenAIMixin` class provides direct OpenAI API functionality for providers that work with OpenAI-compatible endpoints. It includes:

#### Direct API Methods
- **`openai_completion()`**: Legacy text completion API with full parameter support
- **`openai_chat_completion()`**: Chat completion API supporting streaming, tools, and function calling
- **`openai_embeddings()`**: Text embeddings generation with customizable encoding and dimensions

#### Model Management
- **`check_model_availability()`**: Queries the API endpoint to verify if a model exists and is accessible

#### Client Management
- **`client` property**: Automatically creates and configures AsyncOpenAI client instances using your provider's credentials

#### Required Implementation

To use `OpenAIMixin`, your provider must implement these abstract methods:

```python
@abstractmethod
def get_api_key(self) -> str:
    """Return the API key for authentication"""
    pass


@abstractmethod
def get_base_url(self) -> str:
    """Return the OpenAI-compatible API base URL"""
    pass
```

## Testing the Provider

Before running tests, you must have required dependencies installed. This depends on the providers or distributions you are testing. For example, if you are testing the `together` distribution, you should install dependencies via `llama stack build --distro together`.

### 1. Integration Testing

Integration tests are located in {repopath}`tests/integration`. These tests use the python client-SDK APIs (from the `llama_stack_client` package) to test functionality. Since these tests use client APIs, they can be run either by pointing to an instance of the Llama Stack server or "inline" by using `LlamaStackAsLibraryClient`.

Consult {repopath}`tests/integration/README.md` for more details on how to run the tests.

Note that each provider's `sample_run_config()` method (in the configuration class for that provider)
 typically references some environment variables for specifying API keys and the like. You can set these in the environment or pass these via the `--env` flag to the test command.


### 2. Unit Testing

Unit tests are located in {repopath}`tests/unit`. Provider-specific unit tests are located in {repopath}`tests/unit/providers`. These tests are all run automatically as part of the CI process.

Consult {repopath}`tests/unit/README.md` for more details on how to run the tests manually.

### 3. Additional end-to-end testing

1. Start a Llama Stack server with your new provider
2. Verify compatibility with existing client scripts in the [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main) repository
3. Document which scripts are compatible with your provider

## Submitting Your PR

1. Ensure all tests pass
2. Include a comprehensive test plan in your PR summary
3. Document any known limitations or considerations

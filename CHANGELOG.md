# Changelog

# v0.2.20
Published on: 2025-08-29T22:25:32Z

Here are some key changes that are coming as part of this release.

### Build and Environment

- Environment improvements: fixed env var replacement to preserve types.
- Docker stability: fixed container startup failures for Fireworks AI provider.
- Removed absolute paths in build for better portability.

### Features

- UI Enhancements: Implemented file upload and VectorDB creation/configuration directly in UI.
- Vector Store Improvements: Added keyword, vector, and hybrid search inside vector store.
- Added S3 authorization support for file providers.
- SQL Store: Added inequality support to where clause.

### Documentation

- Fixed post-training docs.
- Added Contributor Guidelines for creating Internal vs. External providers.

### Fixes

- Removed unsupported bfcl scoring function.
- Multiple reliability and configuration fixes for providers and environment handling.

### Engineering / Chores

- Cleaner internal development setup with consistent paths.
- Incremental improvements to provider integration and vector store behavior.


### New Contributors
- @omertuc made their first contribution in #3270
- @r3v5 made their first contribution in vector store hybrid search

---

# v0.2.19
Published on: 2025-08-26T22:06:55Z

## Highlights
* feat: Add CORS configuration support for server by @skamenan7 in https://github.com/llamastack/llama-stack/pull/3201
* feat(api): introduce /rerank by @ehhuang in https://github.com/llamastack/llama-stack/pull/2940
* feat: Add S3 Files Provider by @mattf in https://github.com/llamastack/llama-stack/pull/3202


---

# v0.2.18
Published on: 2025-08-20T01:09:27Z

## Highlights
* Add moderations create API
* Hybrid search in Milvus
* Numerous Responses API improvements
* Documentation updates


---

# v0.2.17
Published on: 2025-08-05T01:51:14Z

## Highlights

* feat(tests): introduce inference record/replay to increase test reliability by @ashwinb in https://github.com/meta-llama/llama-stack/pull/2941
* fix(library_client): improve initialization error handling and prevent AttributeError by @mattf in https://github.com/meta-llama/llama-stack/pull/2944
* fix: use OLLAMA_URL to activate Ollama provider in starter by @ashwinb in https://github.com/meta-llama/llama-stack/pull/2963
* feat(UI): adding MVP playground UI by @franciscojavierarceo in https://github.com/meta-llama/llama-stack/pull/2828
* Standardization of errors (@nathan-weinberg)
* feat: Enable DPO training with HuggingFace inline provider by @Nehanth in https://github.com/meta-llama/llama-stack/pull/2825
* chore: rename templates to distributions by @ashwinb in https://github.com/meta-llama/llama-stack/pull/3035


---

# v0.2.16
Published on: 2025-07-28T23:35:23Z

## Highlights

* Automatic model registration for self-hosted providers (ollama and vllm currently). No need for `INFERENCE_MODEL` environment variables which need to be updated, etc.
* Much simplified starter distribution. Most `ENABLE_` env variables are now gone. When you set `VLLM_URL`, the `vllm` provider is auto-enabled. Similar for `MILVUS_URL`, `PGVECTOR_DB`, etc. Check the [run.yaml](https://github.com/meta-llama/llama-stack/blob/main/llama_stack/templates/starter/run.yaml) for more details.
* All tests migrated to pytest now (thanks @Elbehery)
* DPO implementation in the post-training provider (thanks @Nehanth)
* (Huge!) Support for external APIs and providers thereof (thanks @leseb, @cdoern and others). This is a really big deal -- you can now add more APIs completely out of tree and experiment with them before (optionally) wanting to contribute back.
* `inline::vllm` provider is gone thank you very much
* several improvements to OpenAI inference implementations and LiteLLM backend (thanks @mattf)
* Chroma now supports Vector Store API (thanks @franciscojavierarceo).
* Authorization improvements: Vector Store/File APIs now supports access control (thanks @franciscojavierarceo); Telemetry read APIs are gated according to logged-in user's roles.



---

# v0.2.15
Published on: 2025-07-16T03:30:01Z



---

# v0.2.14
Published on: 2025-07-04T16:06:48Z

## Highlights

* Support for Llama Guard 4
* Added Milvus  support to vector-stores API
* Documentation and zero-to-hero updates for latest APIs


---

# v0.2.13
Published on: 2025-06-28T04:28:11Z

## Highlights
* search_mode support in OpenAI vector store API
* Security fixes


---

# v0.2.12
Published on: 2025-06-20T22:52:12Z

## Highlights
* Filter support in file search
* Support auth attributes in inference and response stores


---

# v0.2.11
Published on: 2025-06-17T20:26:26Z

## Highlights
* OpenAI-compatible vector store APIs
* Hybrid Search in Sqlite-vec
* File search tool in Responses API
* Pagination in inference and response stores
* Added `suffix` to completions API for fill-in-the-middle tasks


---

# v0.2.10.1
Published on: 2025-06-06T20:11:02Z

## Highlights
* ChromaDB provider fix


---

# v0.2.10
Published on: 2025-06-05T23:21:45Z

## Highlights

* OpenAI-compatible embeddings API
* OpenAI-compatible Files API
* Postgres support in starter distro
* Enable ingestion of precomputed embeddings
* Full multi-turn support in Responses API
* Fine-grained access control policy


---

# v0.2.9
Published on: 2025-05-30T20:01:56Z

## Highlights
* Added initial streaming support in Responses API
* UI view for Responses
* Postgres inference store support


---

# v0.2.8
Published on: 2025-05-27T21:03:47Z

# Release v0.2.8

## Highlights

* Server-side MCP with auth firewalls now works in the Stack - both for Agents and Responses
* Get chat completions APIs and UI to show chat completions
* Enable keyword search for sqlite-vec


---

# v0.2.7
Published on: 2025-05-16T20:38:10Z

## Highlights

This is a small update. But a couple highlights:

* feat: function tools in OpenAI Responses by @bbrowning in https://github.com/meta-llama/llama-stack/pull/2094, getting closer to ready. Streaming is the next missing piece.
* feat: Adding support for customizing chunk context in RAG insertion and querying by @franciscojavierarceo in https://github.com/meta-llama/llama-stack/pull/2134
* feat: scaffolding for Llama Stack UI by @ehhuang in https://github.com/meta-llama/llama-stack/pull/2149, more to come in the coming releases.


---

# v0.2.6
Published on: 2025-05-12T18:06:52Z



---

# v0.2.5
Published on: 2025-05-04T20:16:49Z



---

# v0.2.4
Published on: 2025-04-29T17:26:01Z

## Highlights

* One-liner to install and run Llama Stack yay! by @reluctantfuturist in https://github.com/meta-llama/llama-stack/pull/1383
* support for NVIDIA NeMo datastore by @raspawar in https://github.com/meta-llama/llama-stack/pull/1852
* (yuge!) Kubernetes authentication by @leseb in https://github.com/meta-llama/llama-stack/pull/1778
* (yuge!) OpenAI Responses API by @bbrowning in https://github.com/meta-llama/llama-stack/pull/1989
* add api.llama provider, llama-guard-4 model by @ashwinb in https://github.com/meta-llama/llama-stack/pull/2058


---

# v0.2.3
Published on: 2025-04-25T22:46:21Z

## Highlights

* OpenAI compatible inference endpoints and client-SDK support. `client.chat.completions.create()` now works.
* significant improvements and functionality added to the nVIDIA distribution
* many improvements to the test verification suite.
* new inference providers: Ramalama, IBM WatsonX
* many improvements to the Playground UI


---

# v0.2.2
Published on: 2025-04-13T01:19:49Z

## Main changes

- Bring Your Own Provider (@leseb) - use out-of-tree provider code to execute the distribution server
- OpenAI compatible inference API in progress (@bbrowning)
- Provider verifications (@ehhuang)
- Many updates and fixes to playground
- Several llama4 related fixes


---

# v0.2.1
Published on: 2025-04-05T23:13:00Z



---

# v0.2.0
Published on: 2025-04-05T19:04:29Z

## Llama 4 Support

Checkout more at https://www.llama.com



---

# v0.1.9
Published on: 2025-03-29T00:52:23Z

### Build and Test Agents
* Agents: Entire document context with attachments
* RAG: Documentation with sqlite-vec faiss comparison
* Getting started: Fixes to getting started notebook.

### Agent Evals and Model Customization
* (**New**) Post-training: Add nemo customizer

### Better Engineering
* Moved sqlite-vec to non-blocking calls
* Don't return a payload on file delete



---

# v0.1.8
Published on: 2025-03-24T01:28:50Z

# v0.1.8 Release Notes

### Build and Test Agents
* Safety: Integrated NVIDIA as a safety provider.
* VectorDB: Added Qdrant as an inline provider.
* Agents: Added support for multiple tool groups in agents.
* Agents: Simplified imports for Agents in client package


### Agent Evals and Model Customization
* Introduced DocVQA and IfEval benchmarks.

### Deploying and Monitoring Agents
* Introduced a Containerfile and image workflow for the Playground.
* Implemented support for Bearer (API Key) authentication.
* Added attribute-based access control for resources.
* Fixes on docker deployments: use --pull always and standardized the default port to 8321
* Deprecated: /v1/inspect/providers use /v1/providers/ instead

### Better Engineering
* Consolidated scripts under the ./scripts directory.
* Addressed mypy violations in various modules.
* Added Dependabot scans for Python dependencies.
* Implemented a scheduled workflow to update the changelog automatically.
* Enforced concurrency to reduce CI loads.


### New Contributors
* @cmodi-meta made their first contribution in https://github.com/meta-llama/llama-stack/pull/1650
* @jeffmaury made their first contribution in https://github.com/meta-llama/llama-stack/pull/1671
* @derekhiggins made their first contribution in https://github.com/meta-llama/llama-stack/pull/1698
* @Bobbins228 made their first contribution in https://github.com/meta-llama/llama-stack/pull/1745

**Full Changelog**: https://github.com/meta-llama/llama-stack/compare/v0.1.7...v0.1.8

---

# v0.1.7
Published on: 2025-03-14T22:30:51Z

## 0.1.7 Release Notes

###  Build and Test Agents
* Inference: ImageType is now refactored to LlamaStackImageType
* Inference: Added tests to measure TTFT
* Inference: Bring back usage metrics
* Agents: Added endpoint for get agent, list agents and list sessions
* Agents: Automated conversion of type hints in client tool for lite llm format
* Agents: Deprecated ToolResponseMessage in agent.resume API
* Added Provider API for listing and inspecting provider info

### Agent Evals and Model Customization
* Eval: Added new eval benchmarks Math 500 and BFCL v3
* Deploy and Monitoring of Agents
* Telemetry: Fix tracing to work across coroutines

###  Better Engineering
* Display code coverage for unit tests
* Updated call sites (inference, tool calls, agents) to move to async non blocking calls
* Unit tests also run on Python 3.11, 3.12, and 3.13
* Added ollama inference to Integration tests CI
* Improved documentation across examples, testing, CLI, updated providers table )




---

# v0.1.6
Published on: 2025-03-08T04:35:08Z

## 0.1.6 Release Notes

### Build and Test Agents
* Inference: Fixed support for inline vllm provider
* (**New**) Agent: Build & Monitor Agent Workflows with Llama Stack + Anthropic's Best Practice [Notebook](https://github.com/meta-llama/llama-stack/blob/main/docs/notebooks/Llama_Stack_Agent_Workflows.ipynb)
* (**New**) Agent: Revamped agent [documentation](https://llama-stack.readthedocs.io/en/latest/building_applications/agent.html) with more details and examples
* Agent: Unify tools and Python SDK Agents API
* Agent: AsyncAgent Python SDK wrapper supporting async client tool calls
* Agent: Support python functions without @client_tool decorator as client tools
* Agent: deprecation for allow_resume_turn flag, and remove need to specify tool_prompt_format
* VectorIO: MilvusDB support added

### Agent Evals and Model Customization
* (**New**) Agent: Llama Stack RAG Lifecycle [Notebook](https://github.com/meta-llama/llama-stack/blob/main/docs/notebooks/Llama_Stack_RAG_Lifecycle.ipynb)
* Eval: Documentation for eval, scoring, adding new benchmarks
* Eval: Distribution template to run benchmarks on llama & non-llama models
* Eval: Ability to register new custom LLM-as-judge scoring functions
* (**New**) Looking for contributors for open benchmarks. See [documentation](https://llama-stack.readthedocs.io/en/latest/references/evals_reference/index.html#open-benchmark-contributing-guide) for details.

### Deploy and Monitoring of Agents
* Better support for different log levels across all components for better monitoring

### Better Engineering
* Enhance OpenAPI spec to include Error types across all APIs
* Moved all tests to /tests and created unit tests to run on each PR
* Removed all dependencies on llama-models repo


---

# v0.1.5.1
Published on: 2025-02-28T22:37:44Z

## 0.1.5.1 Release Notes
* Fixes for security risk in https://github.com/meta-llama/llama-stack/pull/1327 and https://github.com/meta-llama/llama-stack/pull/1328

**Full Changelog**: https://github.com/meta-llama/llama-stack/compare/v0.1.5...v0.1.5.1

---

# v0.1.5
Published on: 2025-02-28T18:14:01Z

## 0.1.5 Release Notes
###  Build Agents
* Inference: Support more non-llama models (openai, anthropic, gemini)
* Inference: Can use the provider's model name in addition to the HF alias
* Inference: Fixed issues with calling tools that weren't specified in the prompt
* RAG: Improved system prompt for RAG and no more need for hard-coded rag-tool calling
* Embeddings: Added support for Nemo retriever embedding models
* Tools: Added support for MCP tools in Ollama Distribution
* Distributions: Added new Groq distribution

### Customize Models
* Save post-trained checkpoint in SafeTensor format to allow Ollama inference provider to use the post-trained model

### Monitor agents
* More comprehensive logging of agent steps including client tools
* Telemetry inputs/outputs are now structured and queryable
* Ability to retrieve agents session, turn, step by ids

### Better Engineering
* Moved executorch Swift code out of this repo into the llama-stack-client-swift repo, similar to kotlin
* Move most logging to use logger instead of prints
* Completed text /chat-completion and /completion tests


---

# v0.1.4
Published on: 2025-02-25T00:02:43Z

## v0.1.4 Release Notes
Here are the key changes coming as part of this release:

### Build and Test Agents
* Inference: Added support for non-llama models
* Inference: Added option to list all downloaded models and remove models
* Agent: Introduce new api agents.resume_turn to include client side tool execution in the same turn
* Agent: AgentConfig introduces new variable “tool_config” that allows for better tool configuration and system prompt overrides
* Agent: Added logging for agent step start and completion times
* Agent: Added support for logging for tool execution metadata
* Embedding: Updated /inference/embeddings to support asymmetric models, truncation and variable sized outputs
* Embedding: Updated embedding models for Ollama, Together, and Fireworks with available defaults
* VectorIO: Improved performance of sqlite-vec using chunked writes
### Agent Evals and Model Customization
* Deprecated api /eval-tasks. Use /eval/benchmark  instead
* Added CPU training support for TorchTune
### Deploy and Monitoring of Agents
* Consistent view of client and server tool calls in telemetry
### Better Engineering
* Made tests more data-driven for consistent evaluation
* Fixed documentation links and improved API reference generation
* Various small fixes for build scripts and system reliability



---

# v0.1.3
Published on: 2025-02-14T20:24:32Z

## v0.1.3 Release

Here are some key changes that are coming as part of this release.

### Build and Test Agents
Streamlined the initial development experience
- Added support for  llama stack run --image-type venv
- Enhanced vector store options with new sqlite-vec provider and improved Qdrant integration
- vLLM improvements for tool calling and logprobs
- Better handling of sporadic code_interpreter tool calls

### Agent Evals
Better benchmarking and Agent performance assessment
- Renamed eval API /eval-task to /benchmarks
- Improved documentation and notebooks for RAG and evals

### Deploy and Monitoring of Agents
Improved production readiness
- Added usage metrics collection for chat completions
- CLI improvements for provider information
- Improved error handling and system reliability
- Better model endpoint handling and accessibility
- Improved signal handling on distro server

### Better Engineering
Infrastructure and code quality improvements
- Faster text-based chat completion tests
- Improved testing for non-streaming agent apis
- Standardized import formatting with ruff linter
- Added conventional commits standard
- Fixed documentation parsing issues


---

# v0.1.2
Published on: 2025-02-07T22:06:49Z

# TL;DR
- Several stabilizations to development flows after the switch to `uv`
- Migrated CI workflows to new OSS repo - [llama-stack-ops](https://github.com/meta-llama/llama-stack-ops)
- Added automated rebuilds for ReadTheDocs
- Llama Stack server supports HTTPS
- Added system prompt overrides support
- Several bug fixes and improvements to documentation (check out Kubernetes deployment guide by @terrytangyuan )


---

# v0.1.1
Published on: 2025-02-02T02:29:24Z

A bunch of small / big improvements everywhere including support for Windows, switching to `uv` and many provider improvements.


---

# v0.1.0
Published on: 2025-01-24T17:47:47Z

We are excited to announce a stable API release of Llama Stack, which enables developers to build RAG applications and Agents using tools and safety shields, monitor and those agents with telemetry, and evaluate the agent with scoring functions.

## Context
GenAI application developers need more than just an LLM - they need to integrate tools, connect with their data sources, establish guardrails, and ground the LLM responses effectively. Currently, developers must piece together various tools and APIs, complicating the development lifecycle and increasing costs. The result is that developers are spending more time on these integrations rather than focusing on the application logic itself. The bespoke coupling of components also makes it challenging to adopt state-of-the-art solutions in the rapidly evolving GenAI space. This is particularly difficult for open models like Llama, as best practices are not widely established in the open.

Llama Stack was created to provide developers with a comprehensive and coherent interface that simplifies AI application development and codifies best practices across the Llama ecosystem. Since our launch in September 2024, we have seen a huge uptick in interest in Llama Stack APIs by both AI developers and from partners building AI services with Llama models. Partners like Nvidia, Fireworks, and Ollama have collaborated with us to develop implementations across various APIs, including inference, memory, and safety.

With Llama Stack, you can easily build a RAG agent which can also search the web, do complex math, and custom tool calling. You can use telemetry to inspect those traces, and convert telemetry into evals datasets. And with Llama Stack’s plugin architecture and prepackage distributions, you choose to run your agent anywhere - in the cloud with our partners, deploy your own environment using virtualenv or Docker, operate locally with Ollama, or even run on mobile devices with our SDKs. Llama Stack offers unprecedented flexibility while also simplifying the developer experience.

## Release
After iterating on the APIs for the last 3 months, today we’re launching a stable release (V1) of the Llama Stack APIs and the corresponding llama-stack server and client packages(v0.1.0). We now have automated tests for providers. These tests make sure that all provider implementations are verified. Developers can now easily and reliably select distributions or providers based on their specific requirements.

There are example standalone apps in llama-stack-apps.


## Key Features of this release

- **Unified API Layer**
  - Inference: Run LLM models
  - RAG: Store and retrieve knowledge for RAG
  - Agents: Build multi-step agentic workflows
  - Tools: Register tools that can be called by the agent
  - Safety: Apply content filtering and safety policies
  - Evaluation: Test model and agent quality
  - Telemetry: Collect and analyze usage data and complex agentic traces
  - Post Training ( Coming Soon ): Fine tune models for specific use cases

- **Rich Provider Ecosystem**
  - Local Development: Meta's Reference, Ollama
  - Cloud: Fireworks, Together, Nvidia, AWS Bedrock, Groq, Cerebras
  - On-premises: Nvidia NIM, vLLM, TGI, Dell-TGI
  - On-device: iOS and Android support

- **Built for Production**
  - Pre-packaged distributions for common deployment scenarios
  - Backwards compatibility across model versions
  - Comprehensive evaluation capabilities
  - Full observability and monitoring

- **Multiple developer interfaces**
  - CLI: Command line interface
  - Python SDK
  - Swift iOS SDK
  - Kotlin Android SDK

- **Sample llama stack applications**
  - Python
  - iOS
  - Android



---

# v0.1.0rc12
Published on: 2025-01-22T22:24:01Z



---

# v0.0.63
Published on: 2024-12-18T07:17:43Z

A small but important bug-fix release to update the URL datatype for the client-SDKs. The issue affected multimodal agentic turns especially.

**Full Changelog**: https://github.com/meta-llama/llama-stack/compare/v0.0.62...v0.0.63

---


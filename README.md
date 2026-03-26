# Llama Stack

[![PyPI version](https://img.shields.io/pypi/v/llama_stack.svg)](https://pypi.org/project/llama_stack/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/llama-stack)](https://pypi.org/project/llama-stack/)
[![Docker Hub - Pulls](https://img.shields.io/docker/pulls/llamastack/distribution-starter)](https://hub.docker.com/u/llamastack)
[![License](https://img.shields.io/pypi/l/llama_stack.svg)](https://github.com/meta-llama/llama-stack/blob/main/LICENSE)
[![Discord](https://img.shields.io/discord/1257833999603335178?color=6A7EC2&logo=discord&logoColor=ffffff)](https://discord.gg/llama-stack)
[![Unit Tests](https://github.com/meta-llama/llama-stack/actions/workflows/unit-tests.yml/badge.svg?branch=main)](https://github.com/meta-llama/llama-stack/actions/workflows/unit-tests.yml?query=branch%3Amain)
[![Integration Tests](https://github.com/meta-llama/llama-stack/actions/workflows/integration-tests.yml/badge.svg?branch=main)](https://github.com/meta-llama/llama-stack/actions/workflows/integration-tests.yml?query=branch%3Amain)

[**Quick Start**](https://llamastack.github.io/docs/getting_started/quickstart) | [**Documentation**](https://llamastack.github.io/docs) | [**Colab Notebook**](./docs/getting_started.ipynb) | [**Discord**](https://discord.gg/llama-stack)

- [Overview](#overview)
- [API Providers & Distributions](#api-providers--distributions)
- [Resources](#resources)
- [Community](#community)

## Overview

Llama Stack defines and standardizes the core building blocks that simplify AI application development. It provides a unified set of APIs with implementations from leading service providers. Get started instantly:

```bash
curl -LsSf https://github.com/llamastack/llama-stack/raw/main/scripts/install.sh | bash
```

- **Unified API layer** for Inference, RAG, Agents, Tools, Safety, Evals.
- **Plugin architecture** supporting local development, on-premises, cloud, and mobile environments.
- **Prepackaged verified distributions** for a one-stop solution in any environment.
- **Multiple developer interfaces** — CLI and SDKs for Python, Typescript, iOS, and Android.
- **Standalone applications** as examples for production-grade AI apps with Llama Stack.

## API Providers & Distributions

Here is a list of the various API providers and available distributions. See the [full list](https://llamastack.github.io/docs/providers) for details, including [External Providers](https://llamastack.github.io/docs/providers/external).

|    API Provider      | Environments | Agents | Inference | VectorIO | Safety | Eval | DatasetIO |
|:--------------------:|:------------:|:------:|:---------:|:--------:|:------:|:----:|:--------:|
|      SambaNova       | Hosted | | ✅ | | ✅ | | |
|       Cerebras       | Hosted | | ✅ | | | | |
|      Fireworks       | Hosted | ✅ | ✅ | ✅ | | | |
|     AWS Bedrock      | Hosted | | ✅ | | ✅ | | |
|       Together       | Hosted | ✅ | ✅ | | ✅ | | |
|         Groq         | Hosted | | ✅ | | | | |
|        Ollama        | Single Node | | ✅ | | | | |
|         TGI          | Hosted/Single Node | | ✅ | | | | |
|      NVIDIA NIM      | Hosted/Single Node | | ✅ | | ✅ | | |
|       ChromaDB       | Hosted/Single Node | | | ✅ | | | |
|        Milvus        | Hosted/Single Node | | | ✅ | | | |
|        Qdrant        | Hosted/Single Node | | | ✅ | | | |
|       Weaviate       | Hosted/Single Node | | | ✅ | | | |
|      SQLite-vec      | Single Node | | | ✅ | | | |
|      PG Vector       | Single Node | | | ✅ | | | |
|  PyTorch ExecuTorch  | On-device iOS | ✅ | ✅ | | | | |
|         vLLM         | Single Node | | ✅ | | | | |
|        OpenAI        | Hosted | | ✅ | | | | |
|      Anthropic       | Hosted | | ✅ | | | | |
|        Gemini        | Hosted | | ✅ | | | | |
|       WatsonX        | Hosted | | ✅ | | | | |
|     HuggingFace      | Single Node | | | | | | ✅ |
|     NVIDIA NEMO      | Hosted | | ✅ | ✅ | | ✅ | ✅ |
|        NVIDIA        | Hosted | | | | | ✅ | ✅ |
|      Infinispan      | Single Node | | | ✅ | | | |

A **distribution** (or "distro") is a pre-configured provider bundle for a specific deployment scenario — start with local Ollama and transition to production without changing your application code.

|               **Distribution**                |                                                                    **Llama Stack Docker**                                                                     |                                                 Start This Distribution                                                  |
|:---------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------:|
|                Starter Distribution                 |           [llamastack/distribution-starter](https://hub.docker.com/repository/docker/llamastack/distribution-starter/general)           |      [Guide](https://llamastack.github.io/docs/distributions/self_hosted_distro/starter)      |
|                Starter Distribution GPU                 |           [llamastack/distribution-starter-cpu](https://hub.docker.com/repository/docker/llamastack/distribution-starter-cpu/general)           |      [Guide](https://llamastack.github.io/docs/distributions/self_hosted_distro/starter)      |
|                   PostgreSQL                  |                [llamastack/distribution-postgres-demo](https://hub.docker.com/repository/docker/llamastack/distribution-postgres-demo/general)                | N/A |
|                Dell                 |           [llamastack/distribution-dell](https://hub.docker.com/repository/docker/llamastack/distribution-dell/general)           |      [Guide](https://llamastack.github.io/docs/distributions/self_hosted_distro/dell)      |

For the full list including Docker images see the [Distributions Overview](https://llamastack.github.io/docs/distributions).

## Resources

Full docs at [llamastack.github.io/docs](https://llamastack.github.io/docs). Example apps at [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main/examples).

- [Quick Start](https://llamastack.github.io/docs/getting_started/quickstart) — start a Llama Stack server
- [Getting Started Notebook](./docs/getting_started.ipynb) — text and vision inference walkthrough
- [Zero-to-Hero Guide](https://github.com/meta-llama/llama-stack/tree/main/docs/zero_to_hero_guide) — key components with code samples
- [Server CLI Reference](https://llamastack.github.io/docs/references/llama_cli_reference) | [Client CLI Reference](https://llamastack.github.io/docs/references/llama_stack_client_cli_reference)
- [Contributing](CONTRIBUTING.md) | [Adding a new API Provider](https://llamastack.github.io/docs/contributing/new_api_provider) | [Release Process](RELEASE_PROCESS.md)

**Client SDKs** — connect to a Llama Stack server in your preferred language:

|  **Language** |  **Client SDK** | **Package** |
| :----: | :----: | :----: |
| Python |  [llama-stack-client-python](https://github.com/meta-llama/llama-stack-client-python) | [![PyPI version](https://img.shields.io/pypi/v/llama_stack_client.svg)](https://pypi.org/project/llama_stack_client/) |
| Swift  | [llama-stack-client-swift](https://github.com/meta-llama/llama-stack-client-swift) | [![Swift Package Index](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fmeta-llama%2Fllama-stack-client-swift%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/meta-llama/llama-stack-client-swift) |
| Typescript   | [llama-stack-client-typescript](https://github.com/meta-llama/llama-stack-client-typescript) | [![NPM version](https://img.shields.io/npm/v/llama-stack-client.svg)](https://npmjs.org/package/llama-stack-client) |
| Kotlin | [llama-stack-client-kotlin](https://github.com/meta-llama/llama-stack-client-kotlin) | [![Maven version](https://img.shields.io/maven-central/v/com.llama.llamastack/llama-stack-client-kotlin)](https://central.sonatype.com/artifact/com.llama.llamastack/llama-stack-client-kotlin) |

> **Note**: We are considering a transition from Stainless to OpenAPI Generator for SDK generation ([#4609](https://github.com/llamastack/llama-stack/issues/4609)). The `client-sdks/openapi/` directory contains the new tooling for local SDK generation.

## Community

We hold regular community calls every Thursday at 09:00 AM PST — see the [Community Event on Discord](https://discord.com/events/1257833999603335178/1413266296748900513) for details.

[![Star History Chart](https://api.star-history.com/svg?repos=meta-llama/llama-stack&type=Date)](https://www.star-history.com/#meta-llama/llama-stack&Date)

Thanks to all our amazing contributors!

<a href="https://github.com/meta-llama/llama-stack/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=meta-llama/llama-stack" alt="Llama Stack contributors" />
</a>

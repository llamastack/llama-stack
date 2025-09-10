# Llama Stack
Welcome to Llama Stack, the open-source framework for building generative AI applications.
```{admonition} Llama 4 is here!
:class: tip

Check out [Getting Started with Llama 4](https://colab.research.google.com/github/meta-llama/llama-stack/blob/main/docs/getting_started_llama4.ipynb)
```
```{admonition} News
:class: tip

Llama Stack {{ llama_stack_version }} is now available! See the {{ llama_stack_version_link }} for more details.
```


## What is Llama Stack?

Llama Stack defines and standardizes the core building blocks needed to bring generative AI applications to market. It provides a unified set of APIs with implementations from leading service providers, enabling seamless transitions between development and production environments. More specifically, it provides

- **Unified API layer** for Inference, RAG, Agents, Tools, Safety, Evals, and Telemetry.
- **Plugin architecture** to support the rich ecosystem of implementations of the different APIs in different environments like local development, on-premises, cloud, and mobile.
- **Prepackaged verified distributions** which offer a one-stop solution for developers to get started quickly and reliably in any environment
- **Multiple developer interfaces** like CLI and SDKs for Python, Node, iOS, and Android
- **Standalone applications** as examples for how to build production-grade AI applications with Llama Stack

```{image} ../_static/llama-stack.png
:alt: Llama Stack
:width: 400px
```

Our goal is to provide pre-packaged implementations (aka "distributions") which can be run in a variety of deployment environments. LlamaStack can assist you in your entire app development lifecycle - start iterating on local, mobile or desktop and seamlessly transition to on-prem or public cloud deployments. At every point in this transition, the same set of APIs and the same developer experience is available.

## How does Llama Stack work?
Llama Stack consists of a [server](./distributions/index.md) (with multiple pluggable API [providers](./providers/index.md)) and Client SDKs (see below) meant to
be used in your applications. The server can be run in a variety of environments, including local (inline)
development, on-premises, and cloud. The client SDKs are available for Python, Swift, Node, and
Kotlin.

## Quick Links

- Ready to build? Check out the [Quick Start](getting_started/index) to get started.
- Want to contribute? See the [Contributing](contributing/index) guide.

## Supported Llama Stack Implementations

A number of "adapters" are available for some popular Inference and Vector Store providers. For other APIs (particularly Safety and Agents), we provide *reference implementations* you can use to get started. We expect this list to grow over time. We are slowly onboarding more providers to the ecosystem as we get more confidence in the APIs.

**Inference API**
|  **Provider** |  **Environments** |
| :----: | :----: |
|  Meta Reference  |  Single Node |
|  Ollama  | Single Node   |
|  Fireworks  |  Hosted  |
|  Together  |  Hosted  |
|  NVIDIA NIM  |  Hosted and Single Node  |
|  vLLM  | Hosted and Single Node |
|  TGI  |  Hosted and Single Node  |
|  AWS Bedrock  |  Hosted  |
|  Cerebras  |  Hosted  |
|  Groq  |  Hosted  |
|  SambaNova  |  Hosted  |
| PyTorch ExecuTorch | On-device iOS, Android |
|  OpenAI  |  Hosted  |
|  Anthropic  |  Hosted  |
|  Gemini  |  Hosted  |
|  WatsonX  |  Hosted  |

**Agents API**
|  **Provider** |  **Environments** |
| :----: | :----: |
|  Meta Reference  |  Single Node |
|  Fireworks  |  Hosted  |
|  Together  |  Hosted  |
|  PyTorch ExecuTorch | On-device iOS |

**Vector IO API**
|  **Provider** |  **Environments** |
| :----: | :----: |
|  FAISS | Single Node |
|  SQLite-Vec | Single Node |
|  Chroma | Hosted and Single Node |
|  Milvus | Hosted and Single Node |
|  Postgres (PGVector) | Hosted and Single Node |
|  Weaviate | Hosted |
|  Qdrant  | Hosted and Single Node |

**Safety API**
|  **Provider** |  **Environments** |
| :----: | :----: |
|  Llama Guard | Depends on Inference Provider |
|  Prompt Guard | Single Node |
|  Code Scanner | Single Node |
|  AWS Bedrock | Hosted |

**Post Training API**
|  **Provider** |  **Environments** |
| :----: | :----: |
|  Meta Reference  |  Single Node |
|  HuggingFace  |  Single Node |
|  TorchTune  |  Single Node |
|  NVIDIA NEMO  |  Hosted |

**Eval API**
|  **Provider** |  **Environments** |
| :----: | :----: |
|  Meta Reference  |  Single Node |
|  NVIDIA NEMO  |  Hosted |

**Telemetry API**
|  **Provider** |  **Environments** |
| :----: | :----: |
|  Meta Reference  |  Single Node |

**Tool Runtime API**
|  **Provider** |  **Environments** |
| :----: | :----: |
|  Brave Search | Hosted |
|  RAG Runtime | Single Node |

```{toctree}
:hidden:
:maxdepth: 3

self
getting_started/index
concepts/index
providers/index
distributions/index
advanced_apis/index
building_applications/index
deploying/index
contributing/index
references/index
```

# Llama Stack

[![PyPI version](https://img.shields.io/pypi/v/llama_stack.svg)](https://pypi.org/project/llama_stack/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/llama-stack)](https://pypi.org/project/llama-stack/)
[![License](https://img.shields.io/pypi/l/llama_stack.svg)](https://github.com/meta-llama/llama-stack/blob/main/LICENSE)
[![Discord](https://img.shields.io/discord/1257833999603335178?color=6A7EC2&logo=discord&logoColor=ffffff)](https://discord.gg/llama-stack)
[![Unit Tests](https://github.com/meta-llama/llama-stack/actions/workflows/unit-tests.yml/badge.svg?branch=main)](https://github.com/meta-llama/llama-stack/actions/workflows/unit-tests.yml?query=branch%3Amain)
[![Integration Tests](https://github.com/meta-llama/llama-stack/actions/workflows/integration-tests.yml/badge.svg?branch=main)](https://github.com/meta-llama/llama-stack/actions/workflows/integration-tests.yml?query=branch%3Amain)

[**Quick Start**](https://llamastack.github.io/docs/getting_started/quickstart) | [**Documentation**](https://llamastack.github.io/docs) | [**Colab Notebook**](./docs/getting_started.ipynb) | [**Discord**](https://discord.gg/llama-stack)


### ✨🎉 Llama 4 Support  🎉✨
We released [Version 0.2.0](https://github.com/meta-llama/llama-stack/releases/tag/v0.2.0) with support for the Llama 4 herd of models released by Meta.

<details>

<summary>👋 Click here to see how to run Llama 4 models on Llama Stack </summary>

\
*Note you need 8xH100 GPU-host to run these models*

```bash
pip install -U llama_stack

MODEL="Llama-4-Scout-17B-16E-Instruct"
# get meta url from llama.com
huggingface-cli download meta-llama/$MODEL --local-dir ~/.llama/$MODEL

# install dependencies for the distribution
llama stack list-deps meta-reference-gpu | xargs -L1 uv pip install

# start a llama stack server
INFERENCE_MODEL=meta-llama/$MODEL llama stack run meta-reference-gpu

# install client to interact with the server
pip install llama-stack-client
```
### CLI
```bash
# Run a chat completion
MODEL="Llama-4-Scout-17B-16E-Instruct"

llama-stack-client --endpoint http://localhost:8321 \
inference chat-completion \
--model-id meta-llama/$MODEL \
--message "write a haiku for meta's llama 4 models"

OpenAIChatCompletion(
    ...
    choices=[
        OpenAIChatCompletionChoice(
            finish_reason='stop',
            index=0,
            message=OpenAIChatCompletionChoiceMessageOpenAIAssistantMessageParam(
                role='assistant',
                content='...**Silent minds awaken,**  \n**Whispers of billions of words,**  \n**Reasoning breaks the night.**  \n\n—  \n*This haiku blends the essence of LLaMA 4\'s capabilities with nature-inspired metaphor, evoking its vast training data and transformative potential.*',
                ...
            ),
            ...
        )
    ],
    ...
)
```
### Python SDK
```python
from llama_stack_client import LlamaStackClient

client = LlamaStackClient(base_url=f"http://localhost:8321")

model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
prompt = "Write a haiku about coding"

print(f"User> {prompt}")
response = client.chat.completions.create(
    model=model_id,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ],
)
print(f"Assistant> {response.choices[0].message.content}")
```
As more providers start supporting Llama 4, you can use them in Llama Stack as well. We are adding to the list. Stay tuned!


</details>

### 🚀 One-Line Installer 🚀

To try Llama Stack locally, run:

```bash
curl -LsSf https://github.com/llamastack/llama-stack/raw/main/scripts/install.sh | bash
```

### Overview

Llama Stack standardizes the core building blocks that simplify AI application development. It codifies best practices across the Llama ecosystem. More specifically, it provides

- **Unified API layer** for Inference, RAG, Agents, Tools, Safety, Evals.
- **Plugin architecture** to support the rich ecosystem of different API implementations in various environments, including local development, on-premises, cloud, and mobile.
- **Prepackaged verified distributions** which offer a one-stop solution for developers to get started quickly and reliably in any environment.
- **Multiple developer interfaces** like CLI and SDKs for Python, Typescript, iOS, and Android.
- **Standalone applications** as examples for how to build production-grade AI applications with Llama Stack.

<div style="text-align: center;">
  <img
    src="https://github.com/user-attachments/assets/33d9576d-95ea-468d-95e2-8fa233205a50"
    width="480"
    title="Llama Stack"
    alt="Llama Stack"
  />
</div>

### Llama Stack Benefits
- **Flexible Options**: Developers can choose their preferred infrastructure without changing APIs and enjoy flexible deployment choices.
- **Consistent Experience**: With its unified APIs, Llama Stack makes it easier to build, test, and deploy AI applications with consistent application behavior.
- **Robust Ecosystem**: Llama Stack is already integrated with distribution partners (cloud providers, hardware vendors, and AI-focused companies) that offer tailored infrastructure, software, and services for deploying Llama models.

By reducing friction and complexity, Llama Stack empowers developers to focus on what they do best: building transformative generative AI applications.

### API Providers
Here is a list of the various API providers and available distributions that can help developers get started easily with Llama Stack.
Please checkout for [full list](https://llamastack.github.io/docs/providers)

| API Provider Builder | Environments | Agents | Inference | VectorIO | Safety | Post Training | Eval | DatasetIO |
|:--------------------:|:------------:|:------:|:---------:|:--------:|:------:|:-------------:|:----:|:--------:|
|    Meta Reference    | Single Node | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
|      SambaNova       | Hosted | | ✅ | | ✅ | | | |
|       Cerebras       | Hosted | | ✅ | | | | | |
|      Fireworks       | Hosted | ✅ | ✅ | ✅ | | | | |
|     AWS Bedrock      | Hosted | | ✅ | | ✅ | | | |
|       Together       | Hosted | ✅ | ✅ | | ✅ | | | |
|         Groq         | Hosted | | ✅ | | | | | |
|        Ollama        | Single Node | | ✅ | | | | | |
|         TGI          | Hosted/Single Node | | ✅ | | | | | |
|      NVIDIA NIM      | Hosted/Single Node | | ✅ | | ✅ | | | |
|       ChromaDB       | Hosted/Single Node | | | ✅ | | | | |
|        Milvus        | Hosted/Single Node | | | ✅ | | | | |
|        Qdrant        | Hosted/Single Node | | | ✅ | | | | |
|       Weaviate       | Hosted/Single Node | | | ✅ | | | | |
|      SQLite-vec      | Single Node | | | ✅ | | | | |
|      PG Vector       | Single Node | | | ✅ | | | | |
|  PyTorch ExecuTorch  | On-device iOS | ✅ | ✅ | | | | | |
|         vLLM         | Single Node | | ✅ | | | | | |
|        OpenAI        | Hosted | | ✅ | | | | | |
|      Anthropic       | Hosted | | ✅ | | | | | |
|        Gemini        | Hosted | | ✅ | | | | | |
|       WatsonX        | Hosted | | ✅ | | | | | |
|     HuggingFace      | Single Node | | | | | ✅ | | ✅ |
|      TorchTune       | Single Node | | | | | ✅ | | |
|     NVIDIA NEMO      | Hosted | | ✅ | ✅ | | ✅ | ✅ | ✅ |
|        NVIDIA        | Hosted | | | | | ✅ | ✅ | ✅ |

> **Note**: Additional providers are available through external packages. See [External Providers](https://llamastack.github.io/docs/providers/external) documentation.

### Distributions

A Llama Stack Distribution (or "distro") is a pre-configured bundle of provider implementations for each API component. Distributions make it easy to get started with a specific deployment scenario - you can begin with a local development setup (eg. ollama) and seamlessly transition to production (eg. Fireworks) without changing your application code.
Here are some of the distributions we support:

|               **Distribution**                |                                                                    **Llama Stack Docker**                                                                     |                                                 Start This Distribution                                                  |
|:---------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------:|
|                Starter Distribution                 |           [llamastack/distribution-starter](https://hub.docker.com/repository/docker/llamastack/distribution-starter/general)           |      [Guide](https://llamastack.github.io/latest/distributions/self_hosted_distro/starter.html)      |
|                Meta Reference                 |           [llamastack/distribution-meta-reference-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-gpu/general)           |      [Guide](https://llamastack.github.io/latest/distributions/self_hosted_distro/meta-reference-gpu.html)      |
|                   PostgreSQL                  |                [llamastack/distribution-postgres-demo](https://hub.docker.com/repository/docker/llamastack/distribution-postgres-demo/general)                |                  |

### Documentation

Please checkout our [Documentation](https://llamastack.github.io/latest/index.html) page for more details.

* CLI references
    * [llama (server-side) CLI Reference](https://llamastack.github.io/latest/references/llama_cli_reference/index.html): Guide for using the `llama` CLI to work with Llama models (download, study prompts), and building/starting a Llama Stack distribution.
    * [llama (client-side) CLI Reference](https://llamastack.github.io/latest/references/llama_stack_client_cli_reference.html): Guide for using the `llama-stack-client` CLI, which allows you to query information about the distribution.
* Getting Started
    * [Quick guide to start a Llama Stack server](https://llamastack.github.io/latest/getting_started/index.html).
    * [Jupyter notebook](./docs/getting_started.ipynb) to walk-through how to use simple text and vision inference llama_stack_client APIs
    * The complete Llama Stack lesson [Colab notebook](https://colab.research.google.com/drive/1dtVmxotBsI4cGZQNsJRYPrLiDeT0Wnwt) of the new [Llama 3.2 course on Deeplearning.ai](https://learn.deeplearning.ai/courses/introducing-multimodal-llama-3-2/lesson/8/llama-stack).
    * A [Zero-to-Hero Guide](https://github.com/meta-llama/llama-stack/tree/main/docs/zero_to_hero_guide) that guide you through all the key components of llama stack with code samples.
* [Contributing](CONTRIBUTING.md)
    * [Adding a new API Provider](https://llamastack.github.io/latest/contributing/new_api_provider.html) to walk-through how to add a new API provider.

### Llama Stack Client SDKs

|  **Language** |  **Client SDK** | **Package** |
| :----: | :----: | :----: |
| Python |  [llama-stack-client-python](https://github.com/meta-llama/llama-stack-client-python) | [![PyPI version](https://img.shields.io/pypi/v/llama_stack_client.svg)](https://pypi.org/project/llama_stack_client/)
| Swift  | [llama-stack-client-swift](https://github.com/meta-llama/llama-stack-client-swift) | [![Swift Package Index](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fmeta-llama%2Fllama-stack-client-swift%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/meta-llama/llama-stack-client-swift)
| Typescript   | [llama-stack-client-typescript](https://github.com/meta-llama/llama-stack-client-typescript) | [![NPM version](https://img.shields.io/npm/v/llama-stack-client.svg)](https://npmjs.org/package/llama-stack-client)
| Kotlin | [llama-stack-client-kotlin](https://github.com/meta-llama/llama-stack-client-kotlin) | [![Maven version](https://img.shields.io/maven-central/v/com.llama.llamastack/llama-stack-client-kotlin)](https://central.sonatype.com/artifact/com.llama.llamastack/llama-stack-client-kotlin)

Check out our client SDKs for connecting to a Llama Stack server in your preferred language, you can choose from [python](https://github.com/meta-llama/llama-stack-client-python), [typescript](https://github.com/meta-llama/llama-stack-client-typescript), [swift](https://github.com/meta-llama/llama-stack-client-swift), and [kotlin](https://github.com/meta-llama/llama-stack-client-kotlin) programming languages to quickly build your applications.

You can find more example scripts with client SDKs to talk with the Llama Stack server in our [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main/examples) repo.


## 🌟 GitHub Star History
## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=meta-llama/llama-stack&type=Date)](https://www.star-history.com/#meta-llama/llama-stack&Date)

## ✨ Contributors

Thanks to all of our amazing contributors!

<a href="https://github.com/meta-llama/llama-stack/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=meta-llama/llama-stack" />
</a>

## Llama Stack architecture

Llama Stack allows you to build different layers of distributions for your AI workloads using various SDKs and API providers.

```{image} ../../_static/llama-stack.png
:alt: Llama Stack
:width: 400px
```

### Benefits of Llama stack

#### Current challenges in custom AI applications

Building production AI applications today requires solving multiple challenges:

**Infrastructure Complexity**

- Running large language models efficiently requires specialized infrastructure.
- Different deployment scenarios (local development, cloud, edge) need different solutions.
- Moving from development to production often requires significant rework.

**Essential Capabilities**

- Safety guardrails and content filtering are necessary in an enterprise setting.
- Just model inference is not enough - Knowledge retrieval and RAG capabilities are required.
- Nearly any application needs composable multi-step workflows.
- Without monitoring, observability and evaluation, you end up operating in the dark.

**Lack of Flexibility and Choice**

- Directly integrating with multiple providers creates tight coupling.
- Different providers have different APIs and abstractions.
- Changing providers requires significant code changes.

#### Our Solution: A Universal Stack

Llama Stack addresses these challenges through a service-oriented, API-first approach:

**Develop Anywhere, Deploy Everywhere**
- Start locally with CPU-only setups
- Move to GPU acceleration when needed
- Deploy to cloud or edge without code changes
- Same APIs and developer experience everywhere

**Production-Ready Building Blocks**
- Pre-built safety guardrails and content filtering
- Built-in RAG and agent capabilities
- Comprehensive evaluation toolkit
- Full observability and monitoring

**True Provider Independence**
- Swap providers without application changes
- Mix and match best-in-class implementations
- Federation and fallback support
- No vendor lock-in

**Robust Ecosystem**
- Llama Stack is already integrated with distribution partners (cloud providers, hardware vendors, and AI-focused companies).
- Ecosystem offers tailored infrastructure, software, and services for deploying a variety of models.


### Our Philosophy

- **Service-Oriented**: REST APIs enforce clean interfaces and enable seamless transitions across different environments.
- **Composability**: Every component is independent but works together seamlessly
- **Production Ready**: Built for real-world applications, not just demos
- **Turnkey Solutions**: Easy to deploy built in solutions for popular deployment scenarios


With Llama Stack, you can focus on building your application while we handle the infrastructure complexity, essential capabilities, and provider integrations.
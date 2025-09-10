## Llama Stack Playground

```{note}
The Llama Stack Playground is currently experimental and subject to change. We welcome feedback and contributions to help improve it.
```

The Llama Stack Playground is an simple interface which aims to:
- Showcase **capabilities** and **concepts** of Llama Stack in an interactive environment
- Demo **end-to-end** application code to help users get started to build their own applications
- Provide an **UI** to help users inspect and understand Llama Stack API providers and resources

### Key Features

#### Playground
Interactive pages for users to play with and explore Llama Stack API capabilities.

##### Chatbot
```{eval-rst}
.. video:: https://github.com/user-attachments/assets/8d2ef802-5812-4a28-96e1-316038c84cbf
    :autoplay:
    :playsinline:
    :muted:
    :loop:
    :width: 100%
```
- **Chat**: Chat with Llama models.
  - This page is a simple chatbot that allows you to chat with Llama models. Under the hood, it uses the `/inference/chat-completion` streaming API to send messages to the model and receive responses.
- **RAG**: Uploading documents to memory_banks and chat with RAG agent
  - This page allows you to upload documents as a `memory_bank` and then chat with a RAG agent to query information about the uploaded documents.
  - Under the hood, it uses Llama Stack's `/agents` API to define and create a RAG agent and chat with it in a session.

##### Evaluations
```{eval-rst}
.. video:: https://github.com/user-attachments/assets/6cc1659f-eba4-49ca-a0a5-7c243557b4f5
    :autoplay:
    :playsinline:
    :muted:
    :loop:
    :width: 100%
```
- **Evaluations (Scoring)**: Run evaluations on your AI application datasets.
  - This page demonstrates the flow evaluation API to run evaluations on your custom AI application datasets. You may upload your own evaluation datasets and run evaluations using available scoring functions.
  - Under the hood, it uses Llama Stack's `/scoring` API to run evaluations on selected scoring functions.

```{eval-rst}
.. video:: https://github.com/user-attachments/assets/345845c7-2a2b-4095-960a-9ae40f6a93cf
    :autoplay:
    :playsinline:
    :muted:
    :loop:
    :width: 100%
```
- **Evaluations (Generation + Scoring)**: Use pre-registered evaluation tasks to evaluate an model or agent candidate
  - This page demonstrates the flow for evaluation API to evaluate an model or agent candidate on pre-defined evaluation tasks. An evaluation task is a combination of dataset and scoring functions.
  - Under the hood, it uses Llama Stack's `/eval` API to run generations and scorings on specified evaluation configs.
  - In order to run this page, you may need to register evaluation tasks and datasets as resources first through the following commands.
  ```bash
    $ llama-stack-client datasets register \
    --dataset-id "mmlu" \
    --provider-id "huggingface" \
    --url "https://huggingface.co/datasets/llamastack/evals" \
    --metadata '{"path": "llamastack/evals", "name": "evals__mmlu__details", "split": "train"}' \
    --schema '{"input_query": {"type": "string"}, "expected_answer": {"type": "string"}, "chat_completion_input": {"type": "string"}}'
    ```

    ```bash
    $ llama-stack-client benchmarks register \
    --eval-task-id meta-reference-mmlu \
    --provider-id meta-reference \
    --dataset-id mmlu \
    --scoring-functions basic::regex_parser_multiple_choice_answer
    ```


##### Inspect
```{eval-rst}
.. video:: https://github.com/user-attachments/assets/01d52b2d-92af-4e3a-b623-a9b8ba22ba99
    :autoplay:
    :playsinline:
    :muted:
    :loop:
    :width: 100%
```
- **API Providers**: Inspect Llama Stack API providers
  - This page allows you to inspect Llama Stack API providers and resources.
  - Under the hood, it uses Llama Stack's `/providers` API to get information about the providers.

- **API Resources**: Inspect Llama Stack API resources
  - This page allows you to inspect Llama Stack API resources (`models`, `datasets`, `memory_banks`, `benchmarks`, `shields`).
  - Under the hood, it uses Llama Stack's `/<resources>/list` API to get information about each resources.
  - Please visit [Core Concepts](../../concepts/index.md) for more details about the resources.

### Starting the Llama Stack Playground

To start the Llama Stack Playground, run the following commands:

1. Start up the Llama Stack API server

```bash
llama stack build --distro together --image-type venv
llama stack run together
```

2. Start Streamlit UI
```bash
uv run --with ".[ui]" streamlit run llama_stack.core/ui/app.py
```

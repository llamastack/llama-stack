# (Experimental) LLama Stack UI

## Docker Setup

:warning: This is a work in progress.

## Developer Setup

1. Start up Llama Stack API server. More details [here](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html).

```
llama stack build --distro together --image-type venv

llama stack run together
```

2. (Optional) Register datasets and eval tasks as resources. If you want to run pre-configured evaluation flows (e.g. Evaluations (Generation + Scoring) Page).

```bash
llama-stack-client datasets register \
--dataset-id "mmlu" \
--provider-id "huggingface" \
--url "https://huggingface.co/datasets/llamastack/evals" \
--metadata '{"path": "llamastack/evals", "name": "evals__mmlu__details", "split": "train"}' \
--schema '{"input_query": {"type": "string"}, "expected_answer": {"type": "string", "chat_completion_input": {"type": "string"}}}'
```

```bash
llama-stack-client benchmarks register \
--eval-task-id meta-reference-mmlu \
--provider-id meta-reference \
--dataset-id mmlu \
--scoring-functions basic::regex_parser_multiple_choice_answer
```

3. Start Streamlit UI

```bash
uv run --with ".[ui]" streamlit run llama_stack.core/ui/app.py
```

## Environment Variables

| Environment Variable       | Description                        | Default Value             |
|----------------------------|------------------------------------|---------------------------|
| LLAMA_STACK_ENDPOINT       | The endpoint for the Llama Stack   | http://localhost:8321     |
| FIREWORKS_API_KEY          | API key for Fireworks provider     | (empty string)            |
| TOGETHER_API_KEY           | API key for Together provider      | (empty string)            |
| SAMBANOVA_API_KEY          | API key for SambaNova provider     | (empty string)            |
| OPENAI_API_KEY             | API key for OpenAI provider        | (empty string)            |

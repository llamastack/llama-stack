---
orphan: true
---
# Meta Reference GPU Distribution

```{toctree}
:maxdepth: 2
:hidden:

self
```

The `llamastack/distribution-{{ name }}` distribution consists of the following provider configurations:

{{ providers_table }}

Note that you need access to nvidia GPUs to run this distribution. This distribution is not compatible with CPU-only machines or machines with AMD GPUs.

{% if run_config_env_vars %}
### Environment Variables

The following environment variables can be configured:

{% for var, (default_value, description) in run_config_env_vars.items() %}
- `{{ var }}`: {{ description }} (default: `{{ default_value }}`)
{% endfor %}
{% endif %}


## Prerequisite: Downloading Models

Please check that you have llama model checkpoints downloaded in `~/.llama` before proceeding. See [installation guide](../../references/llama_cli_reference/download_models.md) here to download the models using the Hugging Face CLI.
```

## Running the Distribution

You can do this via venv or Docker which has a pre-built image.

### Via Docker

This method allows you to get started quickly without having to build the distribution code.

```bash
LLAMA_STACK_PORT=8321
docker run \
  -it \
  --pull always \
  --gpu all \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ~/.llama:/root/.llama \
  -e INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct \
  llamastack/distribution-{{ name }} \
  --port $LLAMA_STACK_PORT
```

If you are using Llama Stack Safety / Shield APIs, use:

```bash
docker run \
  -it \
  --pull always \
  --gpu all \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ~/.llama:/root/.llama \
  -e INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct \
  -e SAFETY_MODEL=meta-llama/Llama-Guard-3-1B \
  llamastack/distribution-{{ name }} \
  --port $LLAMA_STACK_PORT
```

### Via venv

Make sure you have done `uv pip install llama-stack` and have the Llama Stack CLI available.

```bash
llama stack build --distro {{ name }} --image-type venv
INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct \
llama stack run distributions/{{ name }}/run.yaml \
  --port 8321
```

If you are using Llama Stack Safety / Shield APIs, use:

```bash
INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct \
SAFETY_MODEL=meta-llama/Llama-Guard-3-1B \
llama stack run distributions/{{ name }}/run-with-safety.yaml \
  --port 8321
```

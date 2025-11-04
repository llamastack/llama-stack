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

### Via Docker with Custom Run Configuration

You can also run the Docker container with a custom run configuration file by mounting it into the container:

```bash
# Set the path to your custom run.yaml file
CUSTOM_RUN_CONFIG=/path/to/your/custom-run.yaml
LLAMA_STACK_PORT=8321

docker run \
  -it \
  --pull always \
  --gpu all \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ~/.llama:/root/.llama \
  -v $CUSTOM_RUN_CONFIG:/app/custom-run.yaml \
  -e RUN_CONFIG_PATH=/app/custom-run.yaml \
  llamastack/distribution-{{ name }} \
  --port $LLAMA_STACK_PORT
```

**Note**: The run configuration must be mounted into the container before it can be used. The `-v` flag mounts your local file into the container, and the `RUN_CONFIG_PATH` environment variable tells the entrypoint script which configuration to use.

{% if run_configs %}
Available run configurations for this distribution:
{% for config in run_configs %}
- `{{ config }}`
{% endfor %}
{% endif %}

### Via venv

Make sure you have the Llama Stack CLI available.

```bash
llama stack list-deps meta-reference-gpu | xargs -L1 uv pip install
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

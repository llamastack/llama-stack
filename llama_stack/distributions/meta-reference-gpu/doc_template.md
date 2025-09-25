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

Please use `llama model list --downloaded` to check that you have llama model checkpoints downloaded in `~/.llama` before proceeding. See [installation guide](../../references/llama_cli_reference/download_models.md) here to download the models. Run `llama model list` to see the available models to download, and `llama model download` to download the checkpoints.

```
$ llama model list --downloaded
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ Model                                   ┃ Size     ┃ Modified Time       ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ Llama3.2-1B-Instruct:int4-qlora-eo8     │ 1.53 GB  │ 2025-02-26 11:22:28 │
├─────────────────────────────────────────┼──────────┼─────────────────────┤
│ Llama3.2-1B                             │ 2.31 GB  │ 2025-02-18 21:48:52 │
├─────────────────────────────────────────┼──────────┼─────────────────────┤
│ Prompt-Guard-86M                        │ 0.02 GB  │ 2025-02-26 11:29:28 │
├─────────────────────────────────────────┼──────────┼─────────────────────┤
│ Llama3.2-3B-Instruct:int4-spinquant-eo8 │ 3.69 GB  │ 2025-02-26 11:37:41 │
├─────────────────────────────────────────┼──────────┼─────────────────────┤
│ Llama3.2-3B                             │ 5.99 GB  │ 2025-02-18 21:51:26 │
├─────────────────────────────────────────┼──────────┼─────────────────────┤
│ Llama3.1-8B                             │ 14.97 GB │ 2025-02-16 10:36:37 │
├─────────────────────────────────────────┼──────────┼─────────────────────┤
│ Llama3.2-1B-Instruct:int4-spinquant-eo8 │ 1.51 GB  │ 2025-02-26 11:35:02 │
├─────────────────────────────────────────┼──────────┼─────────────────────┤
│ Llama-Guard-3-1B                        │ 2.80 GB  │ 2025-02-26 11:20:46 │
├─────────────────────────────────────────┼──────────┼─────────────────────┤
│ Llama-Guard-3-1B:int4                   │ 0.43 GB  │ 2025-02-26 11:33:33 │
└─────────────────────────────────────────┴──────────┴─────────────────────┘
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
  llamastack/distribution-{{ name }} \
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct
```

If you are using Llama Stack Safety / Shield APIs, use:

```bash
docker run \
  -it \
  --pull always \
  --gpu all \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ~/.llama:/root/.llama \
  llamastack/distribution-{{ name }} \
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct \
  --env SAFETY_MODEL=meta-llama/Llama-Guard-3-1B
```

### Via venv

Make sure you have done `uv pip install llama-stack` and have the Llama Stack CLI available.

```bash
llama stack build --distro {{ name }} --image-type venv
llama stack run distributions/{{ name }}/run.yaml \
  --port 8321 \
  --env INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct
```

If you are using Llama Stack Safety / Shield APIs, use:

```bash
llama stack run distributions/{{ name }}/run-with-safety.yaml \
  --port 8321 \
  --env INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct \
  --env SAFETY_MODEL=meta-llama/Llama-Guard-3-1B
```

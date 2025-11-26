---
orphan: true
---

# Dell Distribution of Llama Stack

```{toctree}
:maxdepth: 2
:hidden:

self
```

The `llamastack/distribution-{{ name }}` distribution consists of the following provider configurations.

{{ providers_table }}

You can use this distribution if you have GPUs and want to run an independent TGI or Dell Enterprise Hub container for running inference.

{% if run_config_env_vars %}
### Environment Variables

The following environment variables can be configured:

{% for var, (default_value, description) in run_config_env_vars.items() %}
- `{{ var }}`: {{ description }} (default: `{{ default_value }}`)
{% endfor %}
{% endif %}


## Setting up Inference server using Dell Enterprise Hub's custom TGI container.

NOTE: This is a placeholder to run inference with TGI. This will be updated to use [Dell Enterprise Hub's containers](https://dell.huggingface.co/authenticated/models) once verified.

```bash
export INFERENCE_PORT=8181
export DEH_URL=http://0.0.0.0:$INFERENCE_PORT
export INFERENCE_MODEL=meta-llama/Llama-3.1-8B-Instruct
export CHROMADB_HOST=localhost
export CHROMADB_PORT=6601
export CHROMA_URL=http://$CHROMADB_HOST:$CHROMADB_PORT
export CUDA_VISIBLE_DEVICES=0
export LLAMA_STACK_PORT=8321

docker run --rm -it \
  --pull always \
  --network host \
  -v $HOME/.cache/huggingface:/data \
  -e HF_TOKEN=$HF_TOKEN \
  -p $INFERENCE_PORT:$INFERENCE_PORT \
  --gpus $CUDA_VISIBLE_DEVICES \
  ghcr.io/huggingface/text-generation-inference \
  --dtype bfloat16 \
  --usage-stats off \
  --sharded false \
  --cuda-memory-fraction 0.7 \
  --model-id $INFERENCE_MODEL \
  --port $INFERENCE_PORT --hostname 0.0.0.0
```

If you are using Llama Stack Safety / Shield APIs, then you will need to also run another instance of a TGI with a corresponding safety model like `meta-llama/Llama-Guard-3-1B` using a script like:

```bash
export SAFETY_INFERENCE_PORT=8282
export DEH_SAFETY_URL=http://0.0.0.0:$SAFETY_INFERENCE_PORT
export SAFETY_MODEL=meta-llama/Llama-Guard-3-1B
export CUDA_VISIBLE_DEVICES=1

docker run --rm -it \
  --pull always \
  --network host \
  -v $HOME/.cache/huggingface:/data \
  -e HF_TOKEN=$HF_TOKEN \
  -p $SAFETY_INFERENCE_PORT:$SAFETY_INFERENCE_PORT \
  --gpus $CUDA_VISIBLE_DEVICES \
  ghcr.io/huggingface/text-generation-inference \
  --dtype bfloat16 \
  --usage-stats off \
  --sharded false \
  --cuda-memory-fraction 0.7 \
  --model-id $SAFETY_MODEL \
  --hostname 0.0.0.0 \
  --port $SAFETY_INFERENCE_PORT
```

## Dell distribution relies on ChromaDB for vector database usage

You can start a chroma-db easily using docker.
```bash
# This is where the indices are persisted
mkdir -p $HOME/chromadb

podman run --rm -it \
  --network host \
  --name chromadb \
  -v $HOME/chromadb:/chroma/chroma \
  -e IS_PERSISTENT=TRUE \
  chromadb/chroma:latest \
  --port $CHROMADB_PORT \
  --host $CHROMADB_HOST
```

## Running Llama Stack

Now you are ready to run Llama Stack with TGI as the inference provider. You can do this via Conda (build code) or Docker which has a pre-built image.

### Via Docker

This method allows you to get started quickly without having to build the distribution code.

```bash
docker run -it \
  --pull always \
  --network host \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v $HOME/.llama:/root/.llama \
  # NOTE: mount the llama-stack directory if testing local changes else not needed
  -v $HOME/git/llama-stack:/app/llama-stack-source \
  # localhost/distribution-dell:dev if building / testing locally
  -e INFERENCE_MODEL=$INFERENCE_MODEL \
  -e DEH_URL=$DEH_URL \
  -e CHROMA_URL=$CHROMA_URL \
  llamastack/distribution-{{ name }}\
  --port $LLAMA_STACK_PORT

```

If you are using Llama Stack Safety / Shield APIs, use:

```bash
# You need a local checkout of llama-stack to run this, get it using
# git clone https://github.com/meta-llama/llama-stack.git
cd /path/to/llama-stack

export SAFETY_INFERENCE_PORT=8282
export DEH_SAFETY_URL=http://0.0.0.0:$SAFETY_INFERENCE_PORT
export SAFETY_MODEL=meta-llama/Llama-Guard-3-1B

docker run \
  -it \
  --pull always \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v $HOME/.llama:/root/.llama \
  -v ./llama_stack/distributions/tgi/run-with-safety.yaml:/root/my-run.yaml \
  -e INFERENCE_MODEL=$INFERENCE_MODEL \
  -e DEH_URL=$DEH_URL \
  -e SAFETY_MODEL=$SAFETY_MODEL \
  -e DEH_SAFETY_URL=$DEH_SAFETY_URL \
  -e CHROMA_URL=$CHROMA_URL \
  llamastack/distribution-{{ name }} \
  --config /root/my-run.yaml \
  --port $LLAMA_STACK_PORT
```

### Via Docker with Custom Run Configuration

You can also run the Docker container with a custom run configuration file by mounting it into the container:

```bash
# Set the path to your custom run.yaml file
CUSTOM_RUN_CONFIG=/path/to/your/custom-run.yaml

docker run -it \
  --pull always \
  --network host \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v $HOME/.llama:/root/.llama \
  -v $CUSTOM_RUN_CONFIG:/app/custom-run.yaml \
  -e RUN_CONFIG_PATH=/app/custom-run.yaml \
  -e INFERENCE_MODEL=$INFERENCE_MODEL \
  -e DEH_URL=$DEH_URL \
  -e CHROMA_URL=$CHROMA_URL \
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

### Via Conda

Make sure you have done `pip install llama-stack` and have the Llama Stack CLI available.

```bash
llama stack list-deps {{ name }} | xargs -L1 pip install
INFERENCE_MODEL=$INFERENCE_MODEL \
DEH_URL=$DEH_URL \
CHROMA_URL=$CHROMA_URL \
llama stack run {{ name }} \
  --port $LLAMA_STACK_PORT
```

If you are using Llama Stack Safety / Shield APIs, use:

```bash
INFERENCE_MODEL=$INFERENCE_MODEL \
DEH_URL=$DEH_URL \
SAFETY_MODEL=$SAFETY_MODEL \
DEH_SAFETY_URL=$DEH_SAFETY_URL \
CHROMA_URL=$CHROMA_URL \
llama stack run ./run-with-safety.yaml \
  --port $LLAMA_STACK_PORT
```

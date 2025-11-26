# llama (server-side) CLI Reference

The `llama` CLI tool helps you set up and use the Llama Stack. The CLI is available on your path after installing the `llama-stack` package.

## Installation

You have two ways to install Llama Stack:

1. **Install as a package**:
   You can install the repository directly from [PyPI](https://pypi.org/project/llama-stack/) by running the following command:
   ```bash
   pip install llama-stack
   ```

2. **Install from source**:
   If you prefer to install from the source code, follow these steps:
   ```bash
    mkdir -p ~/local
    cd ~/local
    git clone git@github.com:meta-llama/llama-stack.git

    uv venv myenv --python 3.12
    source myenv/bin/activate  # On Windows: myenv\Scripts\activate

    cd llama-stack
    pip install -e .


## `llama` subcommands
1. `stack`: Allows you to build a stack using the `llama stack` distribution and run a Llama Stack server. You can read more about how to build a Llama Stack distribution in the [Build your own Distribution](../distributions/building_distro) documentation.

For downloading models, we recommend using the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli). See [Downloading models](#downloading-models) for more information.

### Sample Usage

```
llama --help
```

```
usage: llama [-h] {stack} ...

Welcome to the Llama CLI

options:
  -h, --help  show this help message and exit

subcommands:
  {stack}

  stack                 Operations for the Llama Stack / Distributions
```

## Downloading models

You first need to have models downloaded locally. We recommend using the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) to download models.

First, install the Hugging Face CLI:
```bash
pip install huggingface_hub[cli]
```

Then authenticate and download models:
```bash
# Authenticate with Hugging Face
huggingface-cli login

# Download a model
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --local-dir ~/.llama/Llama-3.2-3B-Instruct
```

## List the downloaded models

To list the downloaded models, you can use the Hugging Face CLI:
```bash
# List all downloaded models in your local cache
huggingface-cli scan-cache
```

# Build your own Distribution


This guide will walk you through the steps to get started with building a Llama Stack distribution from scratch with your choice of API providers.


### Setting your log level

In order to specify the proper logging level users can apply the following environment variable `LLAMA_STACK_LOGGING` with the following format:

`LLAMA_STACK_LOGGING=server=debug;core=info`

Where each category in the following list:

- all
- core
- server
- router
- inference
- agents
- safety
- eval
- tools
- client

Can be set to any of the following log levels:

- debug
- info
- warning
- error
- critical

The default global log level is `info`. `all` sets the log level for all components.

A user can also set `LLAMA_STACK_LOG_FILE` which will pipe the logs to the specified path as well as to the terminal. An example would be: `export LLAMA_STACK_LOG_FILE=server.log`

### Llama Stack Build

In order to build your own distribution, we recommend you clone the `llama-stack` repository.


```
git clone git@github.com:meta-llama/llama-stack.git
cd llama-stack
pip install -e .
```
Use the CLI to build your distribution.
The main points to consider are:
1. **Image Type** - Do you want a venv environment or a Container (eg. Docker)
2. **Template** - Do you want to use a template to build your distribution? or start from scratch ?
3. **Config** - Do you want to use a pre-existing config file to build your distribution?

```
llama stack build -h
usage: llama stack build [-h] [--config CONFIG] [--template TEMPLATE] [--distro DISTRIBUTION] [--list-distros] [--image-type {container,venv}] [--image-name IMAGE_NAME] [--print-deps-only]
                         [--run] [--providers PROVIDERS]

Build a Llama stack container

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to a config file to use for the build. You can find example configs in llama_stack.cores/**/build.yaml. If this argument is not provided, you will be prompted to
                        enter information interactively (default: None)
  --template TEMPLATE   (deprecated) Name of the example template config to use for build. You may use `llama stack build --list-distros` to check out the available distributions (default:
                        None)
  --distro DISTRIBUTION, --distribution DISTRIBUTION
                        Name of the distribution to use for build. You may use `llama stack build --list-distros` to check out the available distributions (default: None)
  --list-distros, --list-distributions
                        Show the available distributions for building a Llama Stack distribution (default: False)
  --image-type {container,venv}
                        Image Type to use for the build. If not specified, will use the image type from the template config. (default: None)
  --image-name IMAGE_NAME
                        [for image-type=container|venv] Name of the virtual environment to use for the build. If not specified, currently active environment will be used if found. (default:
                        None)
  --print-deps-only     Print the dependencies for the stack only, without building the stack (default: False)
  --run                 Run the stack after building using the same image type, name, and other applicable arguments (default: False)
  --providers PROVIDERS
                        Build a config for a list of providers and only those providers. This list is formatted like: api1=provider1,api2=provider2. Where there can be multiple providers per
                        API. (default: None)
```

After this step is complete, a file named `<name>-build.yaml` and template file `<name>-run.yaml` will be generated and saved at the output file path specified at the end of the command.

::::{tab-set}
:::{tab-item} Building from a template
To build from alternative API providers, we provide distribution templates for users to get started building a distribution backed by different providers.

The following command will allow you to see the available templates and their corresponding providers.
```
llama stack build --list-templates
```

```
------------------------------+-----------------------------------------------------------------------------+
| Template Name                | Description                                                                 |
+------------------------------+-----------------------------------------------------------------------------+
| watsonx                      | Use watsonx for running LLM inference                                       |
+------------------------------+-----------------------------------------------------------------------------+
| vllm-gpu                     | Use a built-in vLLM engine for running LLM inference                        |
+------------------------------+-----------------------------------------------------------------------------+
| together                     | Use Together.AI for running LLM inference                                   |
+------------------------------+-----------------------------------------------------------------------------+
| tgi                          | Use (an external) TGI server for running LLM inference                      |
+------------------------------+-----------------------------------------------------------------------------+
| starter                      | Quick start template for running Llama Stack with several popular providers |
+------------------------------+-----------------------------------------------------------------------------+
| sambanova                    | Use SambaNova for running LLM inference and safety                          |
+------------------------------+-----------------------------------------------------------------------------+
| remote-vllm                  | Use (an external) vLLM server for running LLM inference                     |
+------------------------------+-----------------------------------------------------------------------------+
| postgres-demo                | Quick start template for running Llama Stack with several popular providers |
+------------------------------+-----------------------------------------------------------------------------+
| passthrough                  | Use Passthrough hosted llama-stack endpoint for LLM inference               |
+------------------------------+-----------------------------------------------------------------------------+
| open-benchmark               | Distribution for running open benchmarks                                    |
+------------------------------+-----------------------------------------------------------------------------+
| ollama                       | Use (an external) Ollama server for running LLM inference                   |
+------------------------------+-----------------------------------------------------------------------------+
| nvidia                       | Use NVIDIA NIM for running LLM inference, evaluation and safety             |
+------------------------------+-----------------------------------------------------------------------------+
| meta-reference-gpu           | Use Meta Reference for running LLM inference                                |
+------------------------------+-----------------------------------------------------------------------------+
| llama_api                    | Distribution for running e2e tests in CI                                    |
+------------------------------+-----------------------------------------------------------------------------+
| hf-serverless                | Use (an external) Hugging Face Inference Endpoint for running LLM inference |
+------------------------------+-----------------------------------------------------------------------------+
| hf-endpoint                  | Use (an external) Hugging Face Inference Endpoint for running LLM inference |
+------------------------------+-----------------------------------------------------------------------------+
| groq                         | Use Groq for running LLM inference                                          |
+------------------------------+-----------------------------------------------------------------------------+
| fireworks                    | Use Fireworks.AI for running LLM inference                                  |
+------------------------------+-----------------------------------------------------------------------------+
| experimental-post-training   | Experimental template for post training                                     |
+------------------------------+-----------------------------------------------------------------------------+
| dell                         | Dell's distribution of Llama Stack. TGI inference via Dell's custom         |
|                              | container                                                                   |
+------------------------------+-----------------------------------------------------------------------------+
| ci-tests                     | Distribution for running e2e tests in CI                                    |
+------------------------------+-----------------------------------------------------------------------------+
| cerebras                     | Use Cerebras for running LLM inference                                      |
+------------------------------+-----------------------------------------------------------------------------+
| bedrock                      | Use AWS Bedrock for running LLM inference and safety                        |
+------------------------------+-----------------------------------------------------------------------------+
```

You may then pick a template to build your distribution with providers fitted to your liking.

For example, to build a distribution with TGI as the inference provider, you can run:
```
$ llama stack build --distro starter
...
You can now edit ~/.llama/distributions/llamastack-starter/starter-run.yaml and run `llama stack run ~/.llama/distributions/llamastack-starter/starter-run.yaml`
```

```{tip}
The generated `run.yaml` file is a starting point for your configuration. For comprehensive guidance on customizing it for your specific needs, infrastructure, and deployment scenarios, see [Customizing Your run.yaml Configuration](customizing_run_yaml.md).
```
:::
:::{tab-item} Building from Scratch

If the provided templates do not fit your use case, you could start off with running `llama stack build` which will allow you to a interactively enter wizard where you will be prompted to enter build configurations.

It would be best to start with a template and understand the structure of the config file and the various concepts ( APIS, providers, resources, etc.) before starting from scratch.
```
llama stack build

> Enter a name for your Llama Stack (e.g. my-local-stack): my-stack
> Enter the image type you want your Llama Stack to be built as (container or venv): venv

Llama Stack is composed of several APIs working together. Let's select
the provider types (implementations) you want to use for these APIs.

Tip: use <TAB> to see options for the providers.

> Enter provider for API inference: inline::meta-reference
> Enter provider for API safety: inline::llama-guard
> Enter provider for API agents: inline::meta-reference
> Enter provider for API memory: inline::faiss
> Enter provider for API datasetio: inline::meta-reference
> Enter provider for API scoring: inline::meta-reference
> Enter provider for API eval: inline::meta-reference
> Enter provider for API telemetry: inline::meta-reference

 > (Optional) Enter a short description for your Llama Stack:

You can now edit ~/.llama/distributions/llamastack-my-local-stack/my-local-stack-run.yaml and run `llama stack run ~/.llama/distributions/llamastack-my-local-stack/my-local-stack-run.yaml`
```
:::

:::{tab-item} Building from a pre-existing build config file
- In addition to templates, you may customize the build to your liking through editing config files and build from config files with the following command.

- The config file will be of contents like the ones in `llama_stack/distributions/*build.yaml`.

```
llama stack build --config llama_stack/distributions/starter/build.yaml
```
:::

:::{tab-item} Building with External Providers

Llama Stack supports external providers that live outside of the main codebase. This allows you to create and maintain your own providers independently or use community-provided providers.

To build a distribution with external providers, you need to:

1. Configure the `external_providers_dir` in your build configuration file:

```yaml
# Example my-external-stack.yaml with external providers
version: '2'
distribution_spec:
  description: Custom distro for CI tests
  providers:
    inference:
    - remote::custom_ollama
# Add more providers as needed
image_type: container
image_name: ci-test
# Path to external provider implementations
external_providers_dir: ~/.llama/providers.d
```

Here's an example for a custom Ollama provider:

```yaml
adapter:
  adapter_type: custom_ollama
  pip_packages:
  - ollama
  - aiohttp
  - llama-stack-provider-ollama # This is the provider package
  config_class: llama_stack_ollama_provider.config.OllamaImplConfig
  module: llama_stack_ollama_provider
api_dependencies: []
optional_api_dependencies: []
```

The `pip_packages` section lists the Python packages required by the provider, as well as the
provider package itself. The package must be available on PyPI or can be provided from a local
directory or a git repository (git must be installed on the build environment).

2. Build your distribution using the config file:

```
llama stack build --config my-external-stack.yaml
```

For more information on external providers, including directory structure, provider types, and implementation requirements, see the [External Providers documentation](../providers/external.md).
:::

:::{tab-item} Building Container

```{admonition} Podman Alternative
:class: tip

Podman is supported as an alternative to Docker. Set `CONTAINER_BINARY` to `podman` in your environment to use Podman.
```

To build a container image, you may start off from a template and use the `--image-type container` flag to specify `container` as the build image type.

```
llama stack build --distro starter --image-type container
```

```
$ llama stack build --distro starter --image-type container
...
Containerfile created successfully in /tmp/tmp.viA3a3Rdsg/ContainerfileFROM python:3.10-slim
...
```

You can now edit ~/meta-llama/llama-stack/tmp/configs/ollama-run.yaml and run `llama stack run ~/meta-llama/llama-stack/tmp/configs/ollama-run.yaml`
```

Now set some environment variables for the inference model ID and Llama Stack Port and create a local directory to mount into the container's file system.
```
export INFERENCE_MODEL="llama3.2:3b"
export LLAMA_STACK_PORT=8321
mkdir -p ~/.llama
```

After this step is successful, you should be able to find the built container image and test it with the below Docker command:

```
docker run -d \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ~/.llama:/root/.llama \
  localhost/distribution-ollama:dev \
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=$INFERENCE_MODEL \
  --env OLLAMA_URL=http://host.docker.internal:11434
```

Here are the docker flags and their uses:

* `-d`: Runs the container in the detached mode as a background process

* `-p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT`: Maps the container port to the host port for accessing the server

* `-v ~/.llama:/root/.llama`: Mounts the local .llama directory to persist configurations and data

* `localhost/distribution-ollama:dev`: The name and tag of the container image to run

* `--port $LLAMA_STACK_PORT`: Port number for the server to listen on

* `--env INFERENCE_MODEL=$INFERENCE_MODEL`: Sets the model to use for inference

* `--env OLLAMA_URL=http://host.docker.internal:11434`: Configures the URL for the Ollama service

:::

::::


### Running your Stack server
Now, let's start the Llama Stack Distribution Server. You will need the YAML configuration file which was written out at the end by the `llama stack build` step.

```
llama stack run -h
usage: llama stack run [-h] [--port PORT] [--image-name IMAGE_NAME] [--env KEY=VALUE]
                       [--image-type {venv}] [--enable-ui]
                       [config | template]

Start the server for a Llama Stack Distribution. You should have already built (or downloaded) and configured the distribution.

positional arguments:
  config | template     Path to config file to use for the run or name of known template (`llama stack list` for a list). (default: None)

options:
  -h, --help            show this help message and exit
  --port PORT           Port to run the server on. It can also be passed via the env var LLAMA_STACK_PORT. (default: 8321)
  --image-name IMAGE_NAME
                        Name of the image to run. Defaults to the current environment (default: None)
  --env KEY=VALUE       Environment variables to pass to the server in KEY=VALUE format. Can be specified multiple times. (default: None)
  --image-type {venv}
                        Image Type used during the build. This should be venv. (default: None)
  --enable-ui           Start the UI server (default: False)
```

**Note:** Container images built with `llama stack build --image-type container` cannot be run using `llama stack run`. Instead, they must be run directly using Docker or Podman commands as shown in the container building section above.

```
# Start using template name
llama stack run tgi

# Start using config file
llama stack run ~/.llama/distributions/llamastack-my-local-stack/my-local-stack-run.yaml

# Start using a venv
llama stack run --image-type venv ~/.llama/distributions/llamastack-my-local-stack/my-local-stack-run.yaml
```

```
$ llama stack run ~/.llama/distributions/llamastack-my-local-stack/my-local-stack-run.yaml

Serving API inspect
 GET /health
 GET /providers/list
 GET /routes/list
Serving API inference
 POST /inference/chat_completion
 POST /inference/completion
 POST /inference/embeddings
...
Serving API agents
 POST /agents/create
 POST /agents/session/create
 POST /agents/turn/create
 POST /agents/delete
 POST /agents/session/delete
 POST /agents/session/get
 POST /agents/step/get
 POST /agents/turn/get

Listening on ['::', '0.0.0.0']:8321
INFO:     Started server process [2935911]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://['::', '0.0.0.0']:8321 (Press CTRL+C to quit)
INFO:     2401:db00:35c:2d2b:face:0:c9:0:54678 - "GET /models/list HTTP/1.1" 200 OK
```

### Listing Distributions
Using the list command, you can view all existing Llama Stack distributions, including stacks built from templates, from scratch, or using custom configuration files.

```
llama stack list -h
usage: llama stack list [-h]

list the build stacks

options:
  -h, --help  show this help message and exit
```

Example Usage

```
llama stack list
```

```
------------------------------+-----------------------------------------------------------------+--------------+------------+
| Stack Name                  | Path                                                            | Build Config | Run Config |
+------------------------------+-----------------------------------------------------------------------------+--------------+
| together                    | ~/.llama/distributions/together                                 | Yes          | No         |
+------------------------------+-----------------------------------------------------------------------------+--------------+
| bedrock                     | ~/.llama/distributions/bedrock                                  | Yes          | No         |
+------------------------------+-----------------------------------------------------------------------------+--------------+
| starter                     | ~/.llama/distributions/starter                                  | Yes          | Yes        |
+------------------------------+-----------------------------------------------------------------------------+--------------+
| remote-vllm                 | ~/.llama/distributions/remote-vllm                              | Yes          | Yes        |
+------------------------------+-----------------------------------------------------------------------------+--------------+
```

### Removing a Distribution
Use the remove command to delete a distribution you've previously built.

```
llama stack rm -h
usage: llama stack rm [-h] [--all] [name]

Remove the build stack

positional arguments:
  name        Name of the stack to delete (default: None)

options:
  -h, --help  show this help message and exit
  --all, -a   Delete all stacks (use with caution) (default: False)
```

Example
```
llama stack rm llamastack-test
```

To keep your environment organized and avoid clutter, consider using `llama stack list` to review old or unused distributions and `llama stack rm <name>` to delete them when they're no longer needed.

### Troubleshooting

If you encounter any issues, ask questions in our discord or search through our [GitHub Issues](https://github.com/meta-llama/llama-stack/issues), or file an new issue.

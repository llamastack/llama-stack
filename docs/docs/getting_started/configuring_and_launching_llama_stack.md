# Configuring and Launching Llama Stack

This guide walks you through setting up and running Llama Stack, organized from the approach that needs the least infrastructure and knowledge to the most.

## Prerequisites

Before getting started with Llama Stack, you need to have Ollama running locally:

1. **Install and run Ollama**: Follow the [Ollama Getting Started guide](https://ollama.ai/download) to install Ollama on your system.

2. **Verify Ollama is running** at `http://localhost:11434`:
   ```bash
   curl http://localhost:11434
   ```

3. **Set the Ollama URL environment variable**:
   ```bash
   export OLLAMA_URL=http://localhost:11434
   ```

## Method 1: Using Llama Stack CLI (Recommended for Getting Started)

This is the simplest approach that requires minimal infrastructure knowledge. You'll use Python's pip package manager to install and run Llama Stack directly on your machine.

### Step 1: Install Llama Stack

Using pip:
```bash
pip install llama-stack
```

Using uv (alternative):
```bash
# Initialize a new project (if starting fresh)
uv init

# Add llama-stack as a dependency
uv add llama-stack

# Note: If using uv, prefix subsequent commands with 'uv run'
# Example: uv run llama stack build --list-distros
```

### Step 2: Build and Run

The quickest way to get started is to use the starter distribution with a virtual environment:

```bash
llama stack build --distro starter --image-type venv --run
```

This single command will:
- Build a Llama Stack distribution with popular providers
- Create a virtual environment
- Start the server automatically

### Step 3: Verify Installation

Test your Llama Stack server:

#### Basic HTTP Health Checks
```bash
# Check server health
curl http://localhost:8321/health

# List available models
curl http://localhost:8321/v1/models
```

#### Comprehensive Verification (Recommended)
Use the official Llama Stack client for better verification:

```bash
# List all configured providers (recommended)
uv run --with llama-stack-client llama-stack-client providers list

# Alternative if you have llama-stack-client installed
llama-stack-client providers list
```

#### Test Chat Completion
```bash
# Basic HTTP test
curl -X POST http://localhost:8321/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1:8b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Or using the client (more robust)
uv run --with llama-stack-client llama-stack-client inference chat-completion \
  --model llama3.1:8b \
  --message "Hello!"
```

## Method 2: Using Docker or Podman

For users familiar with containerization, Docker provides a consistent environment across different systems with pre-built images.

### Basic Docker Usage

Here's an example for spinning up the Llama Stack server using Docker:

```bash
docker run -it \
  -v ~/.llama:/root/.llama \
  --network=host \
  llamastack/distribution-starter \
  --e OLLAMA_URL=http://localhost:11434
```

### Docker Command Breakdown

- `-it`: Run in interactive mode with TTY allocation
- `-v ~/.llama:/root/.llama`: Mount your local Llama Stack configuration directory
- `--network=host`: Use host networking to access local services like Ollama
- `llamastack/distribution-starter`: The official Llama Stack Docker image
- `--e OLLAMA_URL=http://localhost:11434`: Set environment variable for Ollama URL

### Advanced Docker Configuration

You can customize the Docker deployment with additional environment variables:

```bash
docker run -it \
  -v ~/.llama:/root/.llama \
  -p 8321:8321 \
  -e OLLAMA_URL=http://localhost:11434 \
  -e BRAVE_SEARCH_API_KEY=your_api_key_here \
  -e TAVILY_SEARCH_API_KEY=your_api_key_here \
  llamastack/distribution-starter \
  --port 8321
```

### Environment Variables

Common environment variables you can set:

| Variable | Description | Example |
|----------|-------------|---------|
| `OLLAMA_URL` | URL for Ollama service | `http://localhost:11434` |
| `BRAVE_SEARCH_API_KEY` | API key for Brave search | `your_brave_api_key` |
| `TAVILY_SEARCH_API_KEY` | API key for Tavily search | `your_tavily_api_key` |
| `TOGETHER_API_KEY` | API key for Together AI | `your_together_api_key` |
| `OPENAI_API_KEY` | API key for OpenAI | `your_openai_api_key` |

### Verify Installation

Test your Docker/Podman Llama Stack server:

#### Basic HTTP Health Checks
```bash
# Check server health
curl http://localhost:8321/health

# List available models
curl http://localhost:8321/v1/models
```

#### Comprehensive Verification (Recommended)
Use the official Llama Stack client for better verification:

```bash
# List all configured providers (recommended)
uv run --with llama-stack-client llama-stack-client providers list

# Alternative if you have llama-stack-client installed
llama-stack-client providers list
```

#### Test Chat Completion
```bash
# Basic HTTP test
curl -X POST http://localhost:8321/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1:8b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Or using the client (more robust)
uv run --with llama-stack-client llama-stack-client inference chat-completion \
  --model llama3.1:8b \
  --message "Hello!"
```

## Method 3: Manual Server Configuration and Launch (Advanced)

For complete control over your Llama Stack deployment, you can configure and run the server manually with custom provider selection.

### Step 1: Build a Distribution

Choose a distro and build your Llama Stack distribution:

```bash
# List available distributions
llama stack build --list-distros

# Build with a specific distro
llama stack build --distro watsonx --image-type venv --image-name watsonx-stack

# Or build with a meta-reference distro
llama stack build --distro  meta-reference-gpu --image-type venv --image-name  meta-reference-gpu-stack
```

#### Advanced: Custom Provider Selection (Step 1.a)

If you know the specific providers you want to use, you can supply them directly on the command-line instead of using a pre-built distribution:

```bash
llama stack build --providers inference=remote::ollama,agents=inline::meta-reference,safety=inline::llama-guard,vector_io=inline::faiss,tool_runtime=inline::rag-runtime --image-type venv --image-name custom-stack
```

**Discover Available Options:**

```bash
# List all available APIs
llama stack list-apis

# List all available providers
llama stack list-providers
```

This approach gives you complete control over which providers are included in your stack, allowing for highly customized configurations tailored to your specific needs.

### Select Available Distributions

- **ci-tests**: CI tests for Llama Stack
- **dell**: Dell's distribution of Llama Stack. TGI inference via Dell's custom container
- **meta-reference-gpu**: Use Meta Reference for running LLM inference
- **nvidia**: Use NVIDIA NIM for running LLM inference, evaluation and safety
- **open-benchmark**: Distribution for running open benchmarks
- **postgres-demo**: Quick start template for running Llama Stack with several popular providers
- **starter**: Quick start template for running Llama Stack with several popular providers. This distribution is intended for CPU-only environments
- **starter-gpu**: Quick start template for running Llama Stack with several popular providers. This distribution is intended for GPU-enabled environments
- **watsonx**: Use watsonx for running LLM inference

### Step 2: Configure Your Stack

After building, you can customize the configuration files:

#### Configuration File Locations

- Build config: `~/.llama/distributions/{stack-name}/{stack-name}-build.yaml`
- Runtime config: `~/.llama/distributions/{stack-name}/{stack-name}-run.yaml`

#### Sample Runtime Configuration

```yaml
version: 2

apis:
- inference
- safety
- embeddings
- tool_runtime

providers:
  inference:
  - provider_id: ollama
    provider_type: remote::ollama
    config:
      url: http://localhost:11434

  safety:
  - provider_id: llama-guard
    provider_type: remote::ollama
    config:
      url: http://localhost:11434

  embeddings:
  - provider_id: ollama-embeddings
    provider_type: remote::ollama
    config:
      url: http://localhost:11434

  tool_runtime:
  - provider_id: brave-search
    provider_type: remote::brave-search
    config:
      api_key: ${env.BRAVE_SEARCH_API_KEY:=}
```

### Step 3: Launch the Server

Start your configured Llama Stack server:

```bash
# Run with specific port
llama stack run {stack-name} --port 8321

# Run with environment variables
OLLAMA_URL=http://localhost:11434 llama stack run starter --port 8321

# Run in background
nohup llama stack run starter --port 8321 > llama_stack.log 2>&1 &
```

### Step 4: Verify Installation

Test your Llama Stack server:

#### Basic HTTP Health Checks
```bash
# Check server health
curl http://localhost:8321/health

# List available models
curl http://localhost:8321/v1/models
```

#### Comprehensive Verification (Recommended)
Use the official Llama Stack client for better verification:

```bash
# List all configured providers (recommended)
uv run --with llama-stack-client llama-stack-client providers list

# Alternative if you have llama-stack-client installed
llama-stack-client providers list
```

#### Test Chat Completion
Verify with the client (recommended):

```bash
# Verify providers are configured correctly (recommended first step)
uv run --with llama-stack-client llama-stack-client providers list

# Test chat completion using the client
uv run --with llama-stack-client llama-stack-client inference chat-completion \
  --model llama3.1:8b \
  --message "Hello!"

# Alternative if you have llama-stack-client installed
llama-stack-client providers list
llama-stack-client inference chat-completion \
  --model llama3.1:8b \
  --message "Hello!"

# Or using basic HTTP test
curl -X POST http://localhost:8321/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1:8b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Configuration Management

### Managing Multiple Stacks

You can maintain multiple stack configurations:

```bash
# List all built stacks
llama stack list

# Remove a stack
llama stack rm {stack-name}

# Rebuild with updates
llama stack build --distro starter --image-type venv --image-name starter-v2
```

### Common Configuration Issues

#### Port Conflicts

If port 8321 is already in use:

```bash
# Check what's using the port
netstat -tlnp | grep :8321

# Use a different port
llama stack run starter --port 8322
```

## Troubleshooting

### Common Issues

1. **Docker Permission Denied**:
   ```bash
   sudo docker run -it \
     -v ~/.llama:/root/.llama \
     --network=host \
     llamastack/distribution-starter
   ```

2. **Provider Connection Issues**:
   - Verify external services (Ollama, APIs) are running
   - Check network connectivity and firewall settings
   - Validate API keys and URLs

### Logs and Debugging

Enable detailed logging:

```bash
# Run with debug logging
llama stack run starter --port 8321 --log-level DEBUG

# Check logs in Docker
docker logs <container-id>
```

## Next Steps

Once your Llama Stack server is running:

1. **Explore the APIs**: Test inference, safety, and embeddings endpoints
2. **Integrate with Applications**: Use the server with LangChain, custom applications, or API clients
3. **Scale Your Deployment**: Consider load balancing and high-availability setups
4. **Monitor Performance**: Set up logging and monitoring for production use

For more advanced configurations and production deployments, refer to the [Advanced Configuration Guide](advanced_configuration.md) and [Production Deployment Guide](production_deployment.md).

# Configuring and Launching Llama Stack

This guide walks you through the two primary methods for setting up and running Llama Stack: using Docker containers and configuring the server manually.

## Method 1: Using the Starter Docker Container

The easiest way to get started with Llama Stack is using the pre-built Docker container. This approach eliminates the need for manual dependency management and provides a consistent environment across different systems.

### Prerequisites

- Docker installed and running on your system
- Access to external model providers (e.g., Ollama running locally)

### Basic Docker Usage

Here's an example for spinning up the Llama Stack server using Docker:

```bash
docker run -it \
  -v ~/.llama:/root/.llama \
  --network=host \
  llamastack/distribution-starter \
  --env OLLAMA_URL=http://localhost:11434
```

### Docker Command Breakdown

- `-it`: Run in interactive mode with TTY allocation
- `-v ~/.llama:/root/.llama`: Mount your local Llama Stack configuration directory
- `--network=host`: Use host networking to access local services like Ollama
- `llamastack/distribution-starter`: The official Llama Stack Docker image
- `--env OLLAMA_URL=http://localhost:11434`: Set environment variable for Ollama URL

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

## Method 2: Manual Server Configuration and Launch

For more control over your Llama Stack deployment, you can configure and run the server manually.

### Prerequisites

1. **Install Llama Stack**:
   ```bash
   pip install llama-stack
   ```

2. **Install Provider Dependencies** (as needed):
   ```bash

   # For vector operations
   pip install faiss-cpu

   # For database operations
   pip install sqlalchemy aiosqlite asyncpg
   ```

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

### Select Available Distributions

- **dell**: Dell's distribution for Llama Stack.
- **open-benchmark**: Distribution for running open benchmarks.
- **watsonx**: For IBM Watson integration.
- **starter**: Basic distribution with essential providers.

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

```bash
# Check server health
curl http://localhost:8321/health

# List available models
curl http://localhost:8321/v1/models

# Test chat completion
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

#### Files Provider Missing

If you encounter "files provider not available" errors:

1. Add files API to your configuration:
   ```yaml
   apis:
   - files  # Add this line
   - inference
   - safety
   ```

2. Add files provider:
   ```yaml
   providers:
     files:
     - provider_id: localfs
       provider_type: inline::localfs
       config:
         kvstore:
           type: sqlite
           db_path: ~/.llama/files_store.db
   ```

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

2. **Module Not Found Errors**:
   ```bash
   # Install missing dependencies
   pip install ibm-watsonx-ai faiss-cpu sqlalchemy aiosqlite
   ```

3. **Provider Connection Issues**:
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

# Multiple Llama Stack Servers: Starter Distro Guide

A complete guide to running multiple Llama Stack servers using the **starter distribution** for first-time users.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Verify Llama Stack](#Verify-that-llama-stack-is-installed)
3. [Initialize Starter Distribution](#initialize-starter-distribution)
4. [Set Up Multiple Servers](#set-up-multiple-servers)
5. [Configure API Keys](#configure-api-keys)
6. [Start the Servers](#start-the-servers)
7. [Test Your Setup](#test-your-setup)
8. [Manage Your Servers](#manage-your-servers)
9. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements
- **Operating System**: Linux, macOS, or Windows with WSL2
- **Python**: Version 3.12 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB free space minimum
- **Network**: Stable internet connection

### Check Your System
```bash
# Check Python version
python3 --version

# Check available RAM
free -h

# Check disk space
df -h
```

---

## Verify Llama Stack

### Step 1: Verify that llama stack is installed
```bash
# Verify installation
llama stack --help
```

### Step 2: Initialize Starter Distribution
```bash
# Initialize the starter distribution
llama stack build --template starter --name starter

# This creates ~/.llama/distributions/starter/
```

---

## Set Up Multiple Servers

The starter distribution provides a comprehensive configuration with multiple providers. We'll create **2 servers** based on this starter config:

- **Server 1** (Port 8321): Full starter config with all providers
- **Server 2** (Port 8322): Same config with different database paths (using CLI port override)

### Step 1: Examine the Base Configuration

```bash
# View the starter configuration
cat ~/.llama/distributions/starter/starter-run.yaml
```

### Step 2: Create Server 1 Configuration (Full Starter)

```bash
# Copy the starter config for Server 1
cp ~/.llama/distributions/starter/starter-run.yaml ~/server1-starter.yaml
```

### Step 3: Create Server 2 Configuration (Same Config, Different Databases)

```bash
# Copy starter config for Server 2
cp ~/.llama/distributions/starter/starter-run.yaml ~/server2-starter.yaml

# Change the database paths to avoid conflicts (only change needed!)
sed -i 's|~/.llama/distributions/starter|~/.llama/distributions/starter2|g' ~/server2-starter.yaml
```

### Step 4: Create Separate Database Directories
```bash
# Create separate directories for Server 2
mkdir -p ~/.llama/distributions/starter2
```

**That's it!** No need to modify ports in YAML files - we'll use the CLI `--port` flag instead.

---

## Configure API Keys

The starter configuration supports many providers. Set up the API keys you need:

### Essential API Keys

```bash
# Groq (fast inference)
export GROQ_API_KEY="your_groq_api_key_here"

# OpenAI (if you want to use GPT models)
export OPENAI_API_KEY="your_openai_api_key_here"

# Anthropic (if you want Claude models)
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"

# Ollama (for local models)
export OLLAMA_URL="http://localhost:11434"
```

### Optional API Keys (Set only if you plan to use these providers)

```bash
# Fireworks AI
export FIREWORKS_API_KEY="your_fireworks_api_key"

# Together AI
export TOGETHER_API_KEY="your_together_api_key"

# Gemini
export GEMINI_API_KEY="your_gemini_api_key"

# NVIDIA
export NVIDIA_API_KEY="your_nvidia_api_key"
```

---

## Set Up Ollama (Optional)

If you want to use local models through Ollama:

### Install and Start Ollama

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
```

**macOS:**
```bash
brew install ollama
ollama serve
```

### Download Models (in a new terminal)

```bash
# Download popular models
ollama pull llama3.1:8b
ollama pull llama-guard3:8b
ollama pull all-minilm:l6-v2

# Verify models
ollama list
```

---

## Start the Servers

### Method 1: Run in Separate Terminals (Recommended for Development)

**Terminal 1 - Server 1:**
```bash
cd ~
llama stack run ~/server1-starter.yaml --port 8321
```

**Terminal 2 - Server 2 (Uses CLI port override!):**
```bash
cd ~
llama stack run ~/server2-starter.yaml --port 8322
```

### Method 2: Run in Background

```bash
# Start Server 1 in background
cd ~
nohup llama stack run ~/server1-starter.yaml --port 8321 > server1.log 2>&1 &

# Start Server 2 in background with port override
nohup llama stack run ~/server2-starter.yaml --port 8322 > server2.log 2>&1 &
```

### Method 3: Alternative - Use Environment Variable

```bash
# You can also set port via environment variable
export LLAMA_STACK_PORT=8322
llama stack run ~/server2-starter.yaml

# Or inline
LLAMA_STACK_PORT=8322 llama stack run ~/server2-starter.yaml
```

### Expected Output

Both servers should start successfully:
```
Starting server on port 8321...
Server is running at http://localhost:8321
```

```
Starting server on port 8322...
Server is running at http://localhost:8322
```

---

## Test Your Setup

### Step 1: Health Check

```bash
# Test both servers
curl http://localhost:8321/v1/health
curl http://localhost:8322/v1/health
```

**Expected Response:**
```json
{"status": "OK"}
```

### Step 2: List Available Models

```bash
# Check models on Server 1
curl -s http://localhost:8321/v1/models | python3 -m json.tool

# Check models on Server 2
curl -s http://localhost:8322/v1/models | python3 -m json.tool
```

### Step 3: Test Inference with Different Providers

**Test Groq on Server 1:**
```bash
curl -X POST http://localhost:8321/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello! How are you?"}],
    "model": "groq/llama-3.1-8b-instant"
  }'
```

**Test OpenAI on Server 2 (if you have OpenAI API key):**
```bash
curl -X POST http://localhost:8322/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello from server 2!"}],
    "model": "openai/gpt-4o-mini"
  }'
```

**Test Ollama (if you set it up):**
```bash
curl -X POST http://localhost:8321/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello from Ollama!"}],
    "model": "ollama/llama3.1:8b"
  }'
```

### Step 4: Test Embeddings

```bash
curl -X POST http://localhost:8321/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello world",
    "model": "sentence-transformers/all-MiniLM-L6-v2"
  }'
```

---

## Manage Your Servers

### Check What's Running

```bash
# Check server processes
lsof -i :8321 -i :8322

# Check all llama stack processes
ps aux | grep "llama.*stack"
```

### Stop Servers

**Stop individual servers:**
```bash
# Stop Server 1
kill $(lsof -t -i:8321)

# Stop Server 2
kill $(lsof -t -i:8322)
```

**Stop all servers:**
```bash
pkill -f "llama.*stack.*run"
```

### View Logs (if running in background)

```bash
# Watch Server 1 logs
tail -f server1.log

# Watch Server 2 logs
tail -f server2.log
```

### Restart Servers

```bash
# Stop all servers first
pkill -f "llama.*stack.*run"
sleep 3

# Restart both servers
cd ~
nohup llama stack run ~/server1-starter.yaml > server1.log 2>&1 &
nohup llama stack run ~/server2-starter.yaml > server2.log 2>&1 &
```

---

## Troubleshooting

### Problem: "Port already in use"

```bash
# Find what's using the ports
lsof -i :8321 -i :8322

# Kill processes using the ports
kill $(lsof -t -i:8321)
kill $(lsof -t -i:8322)
```

### Problem: "Provider not available"

The starter config includes many providers that may not have API keys set. This is normal behavior:

```bash
# Check which environment variables are set
env | grep -E "(GROQ|OPENAI|ANTHROPIC|OLLAMA)_"

# Set missing API keys you want to use
export GROQ_API_KEY="your_key_here"
```

### Problem: "No models available"

```bash
# Check available models
curl -s http://localhost:8321/v1/models | python3 -m json.tool

# If empty, check your API keys are set correctly
echo $GROQ_API_KEY
echo $OPENAI_API_KEY
```

### Problem: Ollama connection issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# If not running, start it
ollama serve

# Verify OLLAMA_URL is set
echo $OLLAMA_URL
```

---

## Advanced Usage

### Customize Provider Selection

You can modify the YAML files to enable/disable specific providers:

```yaml
# In your server config, comment out providers you don't want
providers:
  inference:
    # - provider_id: openai       # Disabled
    #   provider_type: remote::openai
    #   config:
    #     api_key: ${env.OPENAI_API_KEY:=}

    - provider_id: groq           # Enabled
      provider_type: remote::groq
      config:
        api_key: ${env.GROQ_API_KEY:=}
```

### You Can Use Different Providers on Different Servers

**Server 1 - Local providers**
- Enable: Ollama, vllm, other local providers
- Disable: OpenAI, Anthropic, Groq, Fireworks

**Server 2 - Remote providers:**
- Enable: OpenAI, Anthropic, Gemini
- Disable: Ollama, vllm and local providers
---

## Summary

You now have **2 Llama Stack servers** running with the starter distribution:

### Server Configuration
- **Server 1**: `http://localhost:8321` (Full starter config)
- **Server 2**: `http://localhost:8322` (Modified starter config)

### Key Files
- `~/server1-starter.yaml` - Server 1 configuration
- `~/server2-starter.yaml` - Server 2 configuration
- `server1.log` - Server 1 logs (if background)
- `server2.log` - Server 2 logs (if background)

### Key Commands
```bash
# Health check
curl http://localhost:8321/v1/health
curl http://localhost:8322/v1/health

# Stop servers
kill $(lsof -t -i:8321)
kill $(lsof -t -i:8322)

# Check processes
lsof -i :8321 -i :8322
```

### Next Steps
1. Create more servers with different configurations if needed.
2. Set up API keys for providers you want to use.
3. Test different models and providers.
4. Customize configurations for your specific needs.
5. Set up monitoring and logging for production use.


---

*This guide uses the official Llama Stack starter distribution for maximum compatibility and feature coverage.*

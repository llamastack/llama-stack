# NVIDIA Safety Provider for LlamaStack

This provider enables safety checks and guardrails for LLM interactions using NVIDIA's NeMo Guardrails service.

## Features

- Run safety checks for messages

## Getting Started

### Prerequisites

- LlamaStack with NVIDIA configuration
- Access to NVIDIA NeMo Guardrails service
- NIM for model to use for safety check is deployed

### Setup

Build the NVIDIA environment:

```bash
uv pip install llama-stack-client
uv run llama stack list-deps nvidia | xargs -L1 uv pip install
```

### Basic Usage using the LlamaStack Python Client

#### Initialize the client

```python
import os

os.environ["NVIDIA_API_KEY"] = "your-api-key"
os.environ["NVIDIA_GUARDRAILS_URL"] = "http://guardrails.test"

from llama_stack.core.library_client import LlamaStackAsLibraryClient

client = LlamaStackAsLibraryClient("nvidia")
client.initialize()
```

#### Create a safety shield

```python
from llama_stack.apis.safety import Shield
from llama_stack.apis.inference import Message

# Create a safety shield
shield = Shield(
    shield_id="your-shield-id",
    provider_resource_id="safety-model-id",  # The model to use for safety checks
    description="Safety checks for content moderation",
)

# Register the shield
await client.safety.register_shield(shield)
```

#### Run safety checks

```python
# Messages to check
messages = [Message(role="user", content="Your message to check")]

# Run safety check
response = await client.safety.run_shield(
    shield_id="your-shield-id",
    messages=messages,
)

# Check for violations
if response.violation:
    print(f"Safety violation detected: {response.violation.user_message}")
    print(f"Violation level: {response.violation.violation_level}")
    print(f"Metadata: {response.violation.metadata}")
else:
    print("No safety violations detected")
```

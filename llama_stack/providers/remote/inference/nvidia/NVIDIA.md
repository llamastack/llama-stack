# NVIDIA Inference Provider for LlamaStack

This provider enables running inference using NVIDIA NIM.

## Features
- Endpoints for completions, chat completions, and embeddings for registered models

## Getting Started

### Prerequisites

- LlamaStack with NVIDIA configuration
- Access to NVIDIA NIM deployment
- NIM for model to use for inference is deployed

### Setup

Build the NVIDIA environment:

```bash
llama stack build --distro nvidia --image-type venv
```

### Basic Usage using the LlamaStack Python Client

#### Initialize the client

```python
import os

os.environ["NVIDIA_API_KEY"] = (
    ""  # Required if using hosted NIM endpoint. If self-hosted, not required.
)
os.environ["NVIDIA_BASE_URL"] = "http://nim.test"  # NIM URL

from llama_stack.core.library_client import LlamaStackAsLibraryClient

client = LlamaStackAsLibraryClient("nvidia")
client.initialize()
```

### Create Completion

> Note on Completion API
>
> The hosted NVIDIA Llama NIMs (e.g., `meta-llama/Llama-3.1-8B-Instruct`) with ```NVIDIA_BASE_URL="https://integrate.api.nvidia.com"``` does not support the ```completion``` method, while the locally deployed NIM does.


```python
response = client.inference.completion(
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    content="Complete the sentence using one word: Roses are red, violets are :",
    stream=False,
    sampling_params={
        "max_tokens": 50,
    },
)
print(f"Response: {response.content}")
```

### Create Chat Completion

```python
response = client.inference.chat_completion(
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {
            "role": "system",
            "content": "You must respond to each message with only one word",
        },
        {
            "role": "user",
            "content": "Complete the sentence using one word: Roses are red, violets are:",
        },
    ],
    stream=False,
    sampling_params={
        "max_tokens": 50,
    },
)
print(f"Response: {response.completion_message.content}")
```

### Tool Calling Example ###
```python
from llama_stack.models.llama.datatypes import ToolDefinition, ToolParamDefinition

tool_definition = ToolDefinition(
    tool_name="get_weather",
    description="Get current weather information for a location",
    parameters={
        "location": ToolParamDefinition(
            param_type="string",
            description="The city and state, e.g. San Francisco, CA",
            required=True,
        ),
        "unit": ToolParamDefinition(
            param_type="string",
            description="Temperature unit (celsius or fahrenheit)",
            required=False,
            default="celsius",
        ),
    },
)

tool_response = client.inference.chat_completion(
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "What's the weather like in San Francisco?"}],
    tools=[tool_definition],
)

print(f"Tool Response: {tool_response.completion_message.content}")
if tool_response.completion_message.tool_calls:
    for tool_call in tool_response.completion_message.tool_calls:
        print(f"Tool Called: {tool_call.tool_name}")
        print(f"Arguments: {tool_call.arguments}")
```

### Structured Output Example
```python
from llama_stack.apis.inference import JsonSchemaResponseFormat, ResponseFormatType

person_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "occupation": {"type": "string"},
    },
    "required": ["name", "age", "occupation"],
}

response_format = JsonSchemaResponseFormat(
    type=ResponseFormatType.json_schema, json_schema=person_schema
)

structured_response = client.inference.chat_completion(
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Create a profile for a fictional person named Alice who is 30 years old and is a software engineer. ",
        }
    ],
    response_format=response_format,
)

print(f"Structured Response: {structured_response.completion_message.content}")
```

### Create Embeddings
> Note on OpenAI embeddings compatibility
>
> NVIDIA asymmetric embedding models (e.g., `nvidia/llama-3.2-nv-embedqa-1b-v2`) require an `input_type` parameter not present in the standard OpenAI embeddings API. The NVIDIA Inference Adapter automatically sets `input_type="query"` when using the OpenAI-compatible embeddings endpoint for NVIDIA. For passage embeddings, use the `embeddings` API with `task_type="document"`.

```python
response = client.inference.embeddings(
    model_id="nvidia/llama-3.2-nv-embedqa-1b-v2",
    contents=["What is the capital of France?"],
    task_type="query",
)
print(f"Embeddings: {response.embeddings}")
```
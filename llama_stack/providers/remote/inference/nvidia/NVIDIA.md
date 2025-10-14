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

### Create Chat Completion

The following example shows how to create a chat completion for an NVIDIA NIM.

```python
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
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
    max_tokens=50,
)
print(f"Response: {response.choices[0].message.content}")
```

### Tool Calling Example ###

The following example shows how to do tool calling for an NVIDIA NIM.

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

tool_response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "What's the weather like in San Francisco?"}],
    tools=[tool_definition],
)

print(f"Tool Response: {tool_response.choices[0].message.content}")
if tool_response.choices[0].message.tool_calls:
    for tool_call in tool_response.choices[0].message.tool_calls:
        print(f"Tool Called: {tool_call.tool_name}")
        print(f"Arguments: {tool_call.arguments}")
```

### Structured Output Example

The following example shows how to do structured output for an NVIDIA NIM.

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

structured_response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Create a profile for a fictional person named Alice who is 30 years old and is a software engineer. ",
        }
    ],
    response_format=response_format,
)

print(f"Structured Response: {structured_response.choices[0].message.content}")
```

### Create Embeddings

The following example shows how to create embeddings for an NVIDIA NIM.

```python
response = client.embeddings.create(
    model="nvidia/llama-3.2-nv-embedqa-1b-v2",
    input=["What is the capital of France?"],
    extra_body={"input_type": "query"},
)
print(f"Embeddings: {response.data}")
```

### Vision Language Models Example

The following example shows how to run vision inference by using an NVIDIA NIM.

```python
def load_image_as_base64(image_path):
    with open(image_path, "rb") as image_file:
        img_bytes = image_file.read()
        return base64.b64encode(img_bytes).decode("utf-8")


image_path = {path_to_the_image}
demo_image_b64 = load_image_as_base64(image_path)

vlm_response = client.chat.completions.create(
    model="nvidia/vila",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": {
                        "data": demo_image_b64,
                    },
                },
                {
                    "type": "text",
                    "text": "Please describe what you see in this image in detail.",
                },
            ],
        }
    ],
)

print(f"VLM Response: {vlm_response.choices[0].message.content}")
```

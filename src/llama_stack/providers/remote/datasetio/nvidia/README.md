# NVIDIA DatasetIO Provider for LlamaStack

This provider enables dataset management using NVIDIA's NeMo Customizer service.

## Features

- Register datasets for fine-tuning LLMs
- Unregister datasets

## Getting Started

### Prerequisites

- LlamaStack with NVIDIA configuration
- Access to Hosted NVIDIA NeMo Microservice
- API key for authentication with the NVIDIA service

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
os.environ["NVIDIA_CUSTOMIZER_URL"] = "http://nemo.test"
os.environ["NVIDIA_DATASET_NAMESPACE"] = "default"
os.environ["NVIDIA_PROJECT_ID"] = "test-project"
from llama_stack.core.library_client import LlamaStackAsLibraryClient

client = LlamaStackAsLibraryClient("nvidia")
client.initialize()
```

#### Register a dataset

```python
client.datasets.register(
    purpose="post-training/messages",
    dataset_id="my-training-dataset",
    source={"type": "uri", "uri": "hf://datasets/default/sample-dataset"},
    metadata={
        "format": "json",
        "description": "Dataset for LLM fine-tuning",
        "provider": "nvidia",
    },
)
```

#### Get a list of all registered datasets

```python
datasets = client.datasets.list()
for dataset in datasets:
    print(f"Dataset ID: {dataset.identifier}")
    print(f"Description: {dataset.metadata.get('description', '')}")
    print(f"Source: {dataset.source.uri}")
    print("---")
```

#### Unregister a dataset

```python
client.datasets.unregister(dataset_id="my-training-dataset")
```

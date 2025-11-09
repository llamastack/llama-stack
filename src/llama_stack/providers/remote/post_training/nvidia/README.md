# NVIDIA Post-Training Provider for LlamaStack

This provider enables fine-tuning of LLMs using NVIDIA's NeMo Customizer service.

## Features

- Supervised fine-tuning of Llama models
- LoRA fine-tuning support
- Job management and status tracking

## Getting Started

### Prerequisites

- LlamaStack with NVIDIA configuration
- Access to Hosted NVIDIA NeMo Customizer service
- Dataset registered in the Hosted NVIDIA NeMo Customizer service
- Base model downloaded and available in the Hosted NVIDIA NeMo Customizer service

### Setup

Build the NVIDIA environment:

```bash
uv pip install llama-stack-client
uv run llama stack list-deps nvidia | xargs -L1 uv pip install
```

### Basic Usage using the LlamaStack Python Client

### Create Customization Job

#### Initialize the client

```python
import os

os.environ["NVIDIA_API_KEY"] = "your-api-key"
os.environ["NVIDIA_CUSTOMIZER_URL"] = "http://nemo.test"
os.environ["NVIDIA_DATASET_NAMESPACE"] = "default"
os.environ["NVIDIA_PROJECT_ID"] = "test-project"
os.environ["NVIDIA_OUTPUT_MODEL_DIR"] = "test-example-model@v1"

from llama_stack.core.library_client import LlamaStackAsLibraryClient

client = LlamaStackAsLibraryClient("nvidia")
client.initialize()
```

#### Configure fine-tuning parameters

```python
from llama_stack_client.types.post_training_supervised_fine_tune_params import (
    TrainingConfig,
    TrainingConfigDataConfig,
    TrainingConfigOptimizerConfig,
)
from llama_stack_client.types.algorithm_config_param import LoraFinetuningConfig
```

#### Set up LoRA configuration

```python
algorithm_config = LoraFinetuningConfig(type="LoRA", adapter_dim=16)
```

#### Configure training data

```python
data_config = TrainingConfigDataConfig(
    dataset_id="your-dataset-id",  # Use client.datasets.list() to see available datasets
    batch_size=16,
)
```

#### Configure optimizer

```python
optimizer_config = TrainingConfigOptimizerConfig(
    lr=0.0001,
)
```

#### Set up training configuration

```python
training_config = TrainingConfig(
    n_epochs=2,
    data_config=data_config,
    optimizer_config=optimizer_config,
)
```

#### Start fine-tuning job

```python
training_job = client.post_training.supervised_fine_tune(
    job_uuid="unique-job-id",
    model="meta-llama/Llama-3.1-8B-Instruct",
    checkpoint_dir="",
    algorithm_config=algorithm_config,
    training_config=training_config,
    logger_config={},
    hyperparam_search_config={},
)
```

### List all jobs

```python
jobs = client.post_training.job.list()
```

###  Check job status

```python
job_status = client.post_training.job.status(job_uuid="your-job-id")
```

### Cancel a job

```python
client.post_training.job.cancel(job_uuid="your-job-id")
```

### Inference with the fine-tuned model

#### 1. Register the model

```python
from llama_stack.apis.models import Model, ModelType

client.models.register(
    model_id="test-example-model@v1",
    provider_id="nvidia",
    provider_model_id="test-example-model@v1",
    model_type=ModelType.llm,
)
```

#### 2. Inference with the fine-tuned model

```python
response = client.completions.create(
    prompt="Complete the sentence using one word: Roses are red, violets are ",
    stream=False,
    model="test-example-model@v1",
    max_tokens=50,
)
print(response.choices[0].text)
```

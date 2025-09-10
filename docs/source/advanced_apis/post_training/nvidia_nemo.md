---
orphan: true
---
# NVIDIA NEMO

[NVIDIA NEMO](https://developer.nvidia.com/nemo-framework) is a remote post training provider for Llama Stack. It provides enterprise-grade fine-tuning capabilities through NVIDIA's NeMo Customizer service.

## Features

- Enterprise-grade fine-tuning capabilities
- Support for LoRA and SFT fine-tuning
- Integration with NVIDIA's NeMo Customizer service
- Support for various NVIDIA-optimized models
- Efficient training with NVIDIA hardware acceleration

## Usage

To use NVIDIA NEMO in your Llama Stack project, follow these steps:

1. Configure your Llama Stack project to use this provider.
2. Set up your NVIDIA API credentials.
3. Kick off a fine-tuning job using the Llama Stack post_training API.

## Setup

You'll need to set the following environment variables:

```bash
export NVIDIA_API_KEY="your-api-key"
export NVIDIA_DATASET_NAMESPACE="default"
export NVIDIA_CUSTOMIZER_URL="your-customizer-url"
export NVIDIA_PROJECT_ID="your-project-id"
export NVIDIA_OUTPUT_MODEL_DIR="your-output-model-dir"
```

## Run Training

You can access the provider and the `supervised_fine_tune` method via the post_training API:

```python
import time
import uuid

from llama_stack_client.types import (
    post_training_supervised_fine_tune_params,
    algorithm_config_param,
)


def create_http_client():
    from llama_stack_client import LlamaStackClient

    return LlamaStackClient(base_url="http://localhost:8321")


client = create_http_client()

# Example Dataset
client.datasets.register(
    purpose="post-training/messages",
    source={
        "type": "uri",
        "uri": "huggingface://datasets/llamastack/simpleqa?split=train",
    },
    dataset_id="simpleqa",
)

training_config = post_training_supervised_fine_tune_params.TrainingConfig(
    data_config=post_training_supervised_fine_tune_params.TrainingConfigDataConfig(
        batch_size=8,  # Default batch size for NEMO
        data_format="instruct",
        dataset_id="simpleqa",
        shuffle=True,
    ),
    n_epochs=50,  # Default epochs for NEMO
    optimizer_config=post_training_supervised_fine_tune_params.TrainingConfigOptimizerConfig(
        lr=0.0001,  # Default learning rate
        weight_decay=0.01,  # NEMO-specific parameter
    ),
    # NEMO-specific parameters
    log_every_n_steps=None,
    val_check_interval=0.25,
    sequence_packing_enabled=False,
    hidden_dropout=None,
    attention_dropout=None,
    ffn_dropout=None,
)

algorithm_config = algorithm_config_param.LoraFinetuningConfig(
    alpha=16,  # Default alpha for NEMO
    type="LoRA",
)

job_uuid = f"test-job{uuid.uuid4()}"

# Example Model - must be a supported NEMO model
training_model = "meta/llama-3.1-8b-instruct"

start_time = time.time()
response = client.post_training.supervised_fine_tune(
    job_uuid=job_uuid,
    logger_config={},
    model=training_model,
    hyperparam_search_config={},
    training_config=training_config,
    algorithm_config=algorithm_config,
    checkpoint_dir="output",
)
print("Job: ", job_uuid)

# Wait for the job to complete!
while True:
    status = client.post_training.job.status(job_uuid=job_uuid)
    if not status:
        print("Job not found")
        break

    print(status)
    if status.status == "completed":
        break

    print("Waiting for job to complete...")
    time.sleep(5)

end_time = time.time()
print("Job completed in", end_time - start_time, "seconds!")

print("Artifacts:")
print(client.post_training.job.artifacts(job_uuid=job_uuid))
```

## Supported Models

Currently supports the following models:
- meta/llama-3.1-8b-instruct
- meta/llama-3.2-1b-instruct

## Supported Parameters

### TrainingConfig
- n_epochs (default: 50)
- data_config
- optimizer_config
- log_every_n_steps
- val_check_interval (default: 0.25)
- sequence_packing_enabled (default: False)
- hidden_dropout (0.0-1.0)
- attention_dropout (0.0-1.0)
- ffn_dropout (0.0-1.0)

### DataConfig
- dataset_id
- batch_size (default: 8)

### OptimizerConfig
- lr (default: 0.0001)
- weight_decay (default: 0.01)

### LoRA Config
- alpha (default: 16)
- type (must be "LoRA")

Note: Some parameters from the standard Llama Stack API are not supported and will be ignored with a warning.

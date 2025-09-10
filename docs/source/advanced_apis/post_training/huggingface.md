---
orphan: true
---
# HuggingFace SFTTrainer

[HuggingFace SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer) is an inline post training provider for Llama Stack. It allows you to run supervised fine tuning on a variety of models using many datasets

## Features

- Simple access through the post_training API
- Fully integrated with Llama Stack
- GPU support, CPU support, and MPS support (MacOS Metal Performance Shaders)

## Usage

To use the HF SFTTrainer in your Llama Stack project, follow these steps:

1. Configure your Llama Stack project to use this provider.
2. Kick off a SFT job using the Llama Stack post_training API.

## Setup

You can access the HuggingFace trainer via the `ollama` distribution:

```bash
llama stack build --distro starter --image-type venv
llama stack run --image-type venv ~/.llama/distributions/ollama/ollama-run.yaml
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
        batch_size=32,
        data_format="instruct",
        dataset_id="simpleqa",
        shuffle=True,
    ),
    gradient_accumulation_steps=1,
    max_steps_per_epoch=0,
    max_validation_steps=1,
    n_epochs=4,
)

algorithm_config = algorithm_config_param.LoraFinetuningConfig(  # this config is also currently mandatory but should not be
    alpha=1,
    apply_lora_to_mlp=True,
    apply_lora_to_output=False,
    lora_attn_modules=["q_proj"],
    rank=1,
    type="LoRA",
)

job_uuid = f"test-job{uuid.uuid4()}"

# Example Model
training_model = "ibm-granite/granite-3.3-8b-instruct"

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

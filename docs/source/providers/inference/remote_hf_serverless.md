# remote::hf::serverless

## Description

HuggingFace Inference API serverless provider for on-demand model inference.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `huggingface_repo` | `<class 'str'>` | No |  | The model ID of the model on the Hugging Face Hub (e.g. 'meta-llama/Meta-Llama-3.1-70B-Instruct') |
| `api_token` | `pydantic.types.SecretStr \| None` | No |  | Your Hugging Face user access token (will default to locally saved token if not provided) |

## Sample Configuration

```yaml
huggingface_repo: ${env.INFERENCE_MODEL}
api_token: ${env.HF_API_TOKEN}

```


# remote::hf::endpoint

## Description

HuggingFace Inference Endpoints provider for dedicated model serving.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `endpoint_name` | `<class 'str'>` | No |  | The name of the Hugging Face Inference Endpoint in the format of '{namespace}/{endpoint_name}' (e.g. 'my-cool-org/meta-llama-3-1-8b-instruct-rce'). Namespace is optional and will default to the user account if not provided. |
| `api_token` | `pydantic.types.SecretStr \| None` | No |  | Your Hugging Face user access token (will default to locally saved token if not provided) |

## Sample Configuration

```yaml
endpoint_name: ${env.INFERENCE_ENDPOINT_NAME}
api_token: ${env.HF_API_TOKEN}

```


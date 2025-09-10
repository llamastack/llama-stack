# remote::together

## Description

Together AI inference provider for open-source models and collaborative AI development.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `allowed_models` | `list[str \| None` | No |  | List of models that should be registered with the model registry. If None, all models are allowed. |
| `url` | `<class 'str'>` | No | https://api.together.xyz/v1 | The URL for the Together AI server |
| `api_key` | `pydantic.types.SecretStr \| None` | No |  | The Together AI API Key |

## Sample Configuration

```yaml
url: https://api.together.xyz/v1
api_key: ${env.TOGETHER_API_KEY:=}

```


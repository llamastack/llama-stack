# remote::sambanova

## Description

SambaNova's safety provider for content moderation and safety filtering.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `url` | `<class 'str'>` | No | https://api.sambanova.ai/v1 | The URL for the SambaNova AI server |
| `api_key` | `pydantic.types.SecretStr \| None` | No |  | The SambaNova cloud API Key |

## Sample Configuration

```yaml
url: https://api.sambanova.ai/v1
api_key: ${env.SAMBANOVA_API_KEY:=}

```


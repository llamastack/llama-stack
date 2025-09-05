# remote::bing-search

## Description

Bing Search tool for web search capabilities using Microsoft's search engine.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `api_key` | `pydantic.types.SecretStr \| None` | No |  |  |
| `top_k` | `<class 'int'>` | No | 3 |  |

## Sample Configuration

```yaml
api_key: ${env.BING_API_KEY:}

```


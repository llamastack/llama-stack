# remote::gemini

## Description

Google Gemini inference provider for accessing Gemini models and Google's AI services.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `api_key` | `str \| None` | No |  | API key for Gemini models |

## Sample Configuration

```yaml
api_key: ${env.GEMINI_API_KEY:=}

```


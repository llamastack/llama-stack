# remote::groq

## Description

Groq inference provider for ultra-fast inference using Groq's LPU technology.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `api_key` | `str \| None` | No |  | The Groq API key |
| `url` | `<class 'str'>` | No | https://api.groq.com | The URL for the Groq AI server |

## Sample Configuration

```yaml
url: https://api.groq.com
api_key: ${env.GROQ_API_KEY:=}

```


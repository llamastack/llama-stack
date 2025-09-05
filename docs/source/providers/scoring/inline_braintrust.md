# inline::braintrust

## Description

Braintrust scoring provider for evaluation and scoring using the Braintrust platform.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `openai_api_key` | `pydantic.types.SecretStr \| None` | No |  | The OpenAI API Key |

## Sample Configuration

```yaml
openai_api_key: ${env.OPENAI_API_KEY:=}

```


# remote::nvidia

## Description

NVIDIA's safety provider for content moderation and safety filtering.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `guardrails_service_url` | `<class 'str'>` | No | http://0.0.0.0:7331 | The url for accessing the Guardrails service |
| `config_id` | `str \| None` | No | self-check | Guardrails configuration ID to use from the Guardrails configuration store |

## Sample Configuration

```yaml
guardrails_service_url: ${env.GUARDRAILS_SERVICE_URL:=http://localhost:7331}
config_id: ${env.NVIDIA_GUARDRAILS_CONFIG_ID:=self-check}

```


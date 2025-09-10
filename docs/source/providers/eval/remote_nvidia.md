# remote::nvidia

## Description

NVIDIA's evaluation provider for running evaluation tasks on NVIDIA's platform.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `evaluator_url` | `<class 'str'>` | No | http://0.0.0.0:7331 | The url for accessing the evaluator service |

## Sample Configuration

```yaml
evaluator_url: ${env.NVIDIA_EVALUATOR_URL:=http://localhost:7331}

```


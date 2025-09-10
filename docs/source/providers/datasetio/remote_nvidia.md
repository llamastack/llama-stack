# remote::nvidia

## Description

NVIDIA's dataset I/O provider for accessing datasets from NVIDIA's data platform.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `api_key` | `str \| None` | No |  | The NVIDIA API key. |
| `dataset_namespace` | `str \| None` | No | default | The NVIDIA dataset namespace. |
| `project_id` | `str \| None` | No | test-project | The NVIDIA project ID. |
| `datasets_url` | `<class 'str'>` | No | http://nemo.test | Base URL for the NeMo Dataset API |

## Sample Configuration

```yaml
api_key: ${env.NVIDIA_API_KEY:=}
dataset_namespace: ${env.NVIDIA_DATASET_NAMESPACE:=default}
project_id: ${env.NVIDIA_PROJECT_ID:=test-project}
datasets_url: ${env.NVIDIA_DATASETS_URL:=http://nemo.test}

```


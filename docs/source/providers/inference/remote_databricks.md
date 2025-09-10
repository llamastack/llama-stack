# remote::databricks

## Description

Databricks inference provider for running models on Databricks' unified analytics platform.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `url` | `<class 'str'>` | No |  | The URL for the Databricks model serving endpoint |
| `api_token` | `<class 'str'>` | No |  | The Databricks API token |

## Sample Configuration

```yaml
url: ${env.DATABRICKS_URL:=}
api_token: ${env.DATABRICKS_API_TOKEN:=}

```


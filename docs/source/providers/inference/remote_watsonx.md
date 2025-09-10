# remote::watsonx

## Description

IBM WatsonX inference provider for accessing AI models on IBM's WatsonX platform.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `url` | `<class 'str'>` | No | https://us-south.ml.cloud.ibm.com | A base url for accessing the watsonx.ai |
| `api_key` | `pydantic.types.SecretStr \| None` | No |  | The watsonx API key, only needed of using the hosted service |
| `project_id` | `str \| None` | No |  | The Project ID key, only needed of using the hosted service |
| `timeout` | `<class 'int'>` | No | 60 | Timeout for the HTTP requests |

## Sample Configuration

```yaml
url: ${env.WATSONX_BASE_URL:=https://us-south.ml.cloud.ibm.com}
api_key: ${env.WATSONX_API_KEY:=}
project_id: ${env.WATSONX_PROJECT_ID:=}

```


# remote::nvidia

## Description

NVIDIA inference provider for accessing NVIDIA NIM models and AI services.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `url` | `<class 'str'>` | No | https://integrate.api.nvidia.com | A base url for accessing the NVIDIA NIM |
| `api_key` | `pydantic.types.SecretStr \| None` | No |  | The NVIDIA API key, only needed of using the hosted service |
| `timeout` | `<class 'int'>` | No | 60 | Timeout for the HTTP requests |
| `append_api_version` | `<class 'bool'>` | No | True | When set to false, the API version will not be appended to the base_url. By default, it is true. |

## Sample Configuration

```yaml
url: ${env.NVIDIA_BASE_URL:=https://integrate.api.nvidia.com}
api_key: ${env.NVIDIA_API_KEY:=}
append_api_version: ${env.NVIDIA_APPEND_API_VERSION:=True}

```


# remote::vllm

## Description

Remote vLLM inference provider for connecting to vLLM servers.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `url` | `str \| None` | No |  | The URL for the vLLM model serving endpoint |
| `max_tokens` | `<class 'int'>` | No | 4096 | Maximum number of tokens to generate. |
| `api_token` | `str \| None` | No | fake | The API token |
| `tls_verify` | `bool \| str` | No | True | Whether to verify TLS certificates. Can be a boolean or a path to a CA certificate file. |
| `refresh_models` | `<class 'bool'>` | No | False | Whether to refresh models periodically |

## Sample Configuration

```yaml
url: ${env.VLLM_URL:=}
max_tokens: ${env.VLLM_MAX_TOKENS:=4096}
api_token: ${env.VLLM_API_TOKEN:=fake}
tls_verify: ${env.VLLM_TLS_VERIFY:=true}

```


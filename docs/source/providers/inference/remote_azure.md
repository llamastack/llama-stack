# remote::azure

## Description


Azure OpenAI inference provider for accessing GPT models and other Azure services.
Provider documentation
https://learn.microsoft.com/en-us/azure/ai-foundry/openai/overview


## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `api_key` | `<class 'pydantic.types.SecretStr'>` | No |  | Azure API key for Azure |
| `api_base` | `<class 'pydantic.networks.HttpUrl'>` | No |  | Azure API base for Azure (e.g., https://your-resource-name.openai.azure.com) |
| `api_version` | `str \| None` | No |  | Azure API version for Azure (e.g., 2024-12-01-preview) |
| `api_type` | `str \| None` | No | azure | Azure API type for Azure (e.g., azure) |

## Sample Configuration

```yaml
api_key: ${env.AZURE_API_KEY:=}
api_base: ${env.AZURE_API_BASE:=}
api_version: ${env.AZURE_API_VERSION:=}
api_type: ${env.AZURE_API_TYPE:=}

```


# remote::tavily-search

## Description

Tavily Search tool for AI-optimized web search with structured results.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `api_key` | `str \| None` | No |  | The Tavily Search API Key |
| `max_results` | `<class 'int'>` | No | 3 | The maximum number of results to return |
| `timeout` | `<class 'float'>` | No | 30.0 | HTTP request timeout for the API |
| `connect_timeout` | `<class 'float'>` | No | 10.0 | HTTP connection timeout in seconds for the API |

## Sample Configuration

```yaml
api_key: ${env.TAVILY_SEARCH_API_KEY:=}
max_results: 3
timeout: 30.0
connect_timeout: 10.0

```


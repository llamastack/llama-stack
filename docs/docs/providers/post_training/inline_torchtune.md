# inline::torchtune

## Description

TorchTune-based post-training provider for fine-tuning and optimizing models using Meta's TorchTune framework.

## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `torch_seed` | `int \| None` | No |  |  |
| `checkpoint_format` | `Literal['meta', 'huggingface'` | No | meta |  |

## Sample Configuration

```yaml
checkpoint_format: meta

```


# Scoring

The Scoring API in Llama Stack allows you to evaluate outputs of your GenAI system using various scoring functions and metrics. This section covers all available scoring providers and their configuration.

## Overview

Llama Stack provides multiple scoring providers:

- **Basic** (`inline::basic`) - Simple evaluation metrics and scoring functions
- **Braintrust** (`inline::braintrust`) - Advanced evaluation using the Braintrust platform
- **LLM-as-Judge** (`inline::llm-as-judge`) - Uses language models to evaluate responses

The Scoring API is associated with `ScoringFunction` resources and provides a suite of out-of-the-box scoring functions. You can also add custom evaluators to meet specific evaluation needs.

## Basic Scoring

Basic scoring provider for simple evaluation metrics and scoring functions. This provider offers fundamental scoring capabilities without external dependencies.

### Configuration

No configuration required - this provider works out of the box.

```yaml
{}
```

### Features

- Simple evaluation metrics (accuracy, precision, recall, F1-score)
- String matching and similarity metrics
- Basic statistical scoring functions
- No external dependencies required
- Fast execution for standard metrics

### Use Cases

- Quick evaluation of basic accuracy metrics
- String similarity comparisons
- Statistical analysis of model outputs
- Development and testing scenarios

## Braintrust

Braintrust scoring provider for evaluation and scoring using the [Braintrust platform](https://braintrustdata.com/). Braintrust provides advanced evaluation capabilities and experiment tracking.

### Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `openai_api_key` | `str \| None` | No |  | The OpenAI API Key for LLM-powered evaluations |

### Sample Configuration

```yaml
openai_api_key: ${env.OPENAI_API_KEY:=}
```

### Features

- Advanced evaluation metrics
- Experiment tracking and comparison
- LLM-powered evaluation functions
- Integration with Braintrust's evaluation suite
- Detailed scoring analytics and insights

### Use Cases

- Production evaluation pipelines
- A/B testing of model versions
- Advanced scoring with custom metrics
- Detailed evaluation reporting and analysis

## LLM-as-Judge

LLM-as-judge scoring provider that uses language models to evaluate and score responses. This approach leverages the reasoning capabilities of large language models to assess quality, relevance, and other subjective metrics.

### Configuration

No configuration required - this provider works out of the box.

```yaml
{}
```

### Features

- Subjective quality evaluation using LLMs
- Flexible evaluation criteria definition
- Natural language evaluation explanations
- Support for complex evaluation scenarios
- Contextual understanding of responses

### Use Cases

- Evaluating response quality and relevance
- Assessing creativity and coherence
- Subjective metric evaluation
- Human-like judgment for complex tasks

## Usage Examples

### Basic Scoring Example

```python
from llama_stack_client import LlamaStackClient

client = LlamaStackClient(base_url="http://localhost:8321")

# Register a basic accuracy scoring function
client.scoring_functions.register(
    scoring_function_id="basic_accuracy",
    provider_id="basic",
    provider_scoring_function_id="accuracy"
)

# Use the scoring function
result = client.scoring.score(
    input_rows=[
        {"expected": "Paris", "actual": "Paris"},
        {"expected": "London", "actual": "Paris"}
    ],
    scoring_function_id="basic_accuracy"
)
print(f"Accuracy: {result.results[0].score}")
```

### LLM-as-Judge Example

```python
# Register an LLM-as-judge scoring function
client.scoring_functions.register(
    scoring_function_id="quality_judge",
    provider_id="llm_judge",
    provider_scoring_function_id="response_quality",
    params={
        "criteria": "Evaluate response quality, relevance, and helpfulness",
        "scale": "1-10"
    }
)

# Score responses using LLM judgment
result = client.scoring.score(
    input_rows=[{
        "query": "What is machine learning?",
        "response": "Machine learning is a subset of AI that enables computers to learn patterns from data..."
    }],
    scoring_function_id="quality_judge"
)
```

### Braintrust Integration Example

```python
# Register a Braintrust scoring function
client.scoring_functions.register(
    scoring_function_id="braintrust_eval",
    provider_id="braintrust",
    provider_scoring_function_id="semantic_similarity"
)

# Run evaluation with Braintrust
result = client.scoring.score(
    input_rows=[{
        "reference": "The capital of France is Paris",
        "candidate": "Paris is the capital city of France"
    }],
    scoring_function_id="braintrust_eval"
)
```

## Best Practices

- **Choose appropriate providers**: Use Basic for simple metrics, Braintrust for advanced analytics, LLM-as-Judge for subjective evaluation
- **Define clear criteria**: When using LLM-as-Judge, provide specific evaluation criteria and scales
- **Validate scoring functions**: Test your scoring functions with known examples before production use
- **Monitor performance**: Track scoring performance and adjust thresholds based on results
- **Combine multiple metrics**: Use different scoring providers together for comprehensive evaluation

## Integration with Evaluation

The Scoring API works closely with the [Evaluation](./evaluation.mdx) API to provide comprehensive evaluation workflows:

1. **Datasets** are loaded via the DatasetIO API
2. **Evaluation** generates model outputs using the Eval API
3. **Scoring** evaluates the quality of outputs using various scoring functions
4. **Results** are aggregated and reported for analysis

## Next Steps

- Check out the [Evaluation](./evaluation.mdx) guide for running complete evaluations
- See the [Building Applications - Evaluation](../building_applications/evals.mdx) guide for application examples
- Review the [Evaluation Reference](../references/evals_reference/) for comprehensive scoring function usage
- Explore the [Evaluation Concepts](../concepts/evaluation_concepts) for detailed conceptual information

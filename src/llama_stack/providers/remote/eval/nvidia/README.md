# NVIDIA NeMo Evaluator Eval Provider


## Overview

For the first integration, Benchmarks are mapped to Evaluation Configs on in the NeMo Evaluator. The full evaluation config object is provided as part of the meta-data. The `dataset_id` and `scoring_functions` are not used.

Below are a few examples of how to register a benchmark, which in turn will create an evaluation config in NeMo Evaluator and how to trigger an evaluation.

### Example for register an academic benchmark

```
POST /eval/benchmarks
```
```json
{
  "benchmark_id": "mmlu",
  "dataset_id": "",
  "scoring_functions": [],
  "metadata": {
    "type": "mmlu"
  }
}
```

### Example for register a custom evaluation

```
POST /eval/benchmarks
```
```json
{
  "benchmark_id": "my-custom-benchmark",
  "dataset_id": "",
  "scoring_functions": [],
  "metadata": {
    "type": "custom",
    "params": {
      "parallelism": 8
    },
    "tasks": {
      "qa": {
        "type": "completion",
        "params": {
          "template": {
            "prompt": "{{prompt}}",
            "max_tokens": 200
          }
        },
        "dataset": {
          "files_url": "hf://datasets/default/sample-basic-test/testing/testing.jsonl"
        },
        "metrics": {
          "bleu": {
            "type": "bleu",
            "params": {
              "references": [
                "{{ideal_response}}"
              ]
            }
          }
        }
      }
    }
  }
}
```

### Example for triggering a benchmark/custom evaluation

```
POST /eval/benchmarks/{benchmark_id}/jobs
```
```json
{
  "benchmark_id": "my-custom-benchmark",
  "benchmark_config": {
    "eval_candidate": {
      "type": "model",
      "model": "meta-llama/Llama3.1-8B-Instruct",
      "sampling_params": {
        "max_tokens": 100,
        "temperature": 0.7
      }
    },
    "scoring_params": {}
  }
}
```

Response example:
```json
{
    "job_id": "eval-1234",
    "status": "in_progress"
}
```

### Example for getting the status of a job
```
GET /eval/benchmarks/{benchmark_id}/jobs/{job_id}
```

Response example:
```json
{
  "job_id": "eval-1234",
  "status": "in_progress"
}
```

### Example for cancelling a job
```
POST /eval/benchmarks/{benchmark_id}/jobs/{job_id}/cancel
```

### Example for getting the results
```
GET /eval/benchmarks/{benchmark_id}/results
```
```json
{
  "generations": [],
  "scores": {
    "{benchmark_id}": {
      "score_rows": [],
      "aggregated_results": {
        "tasks": {},
        "groups": {}
      }
    }
  }
}
```

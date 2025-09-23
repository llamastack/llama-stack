## Evaluation Concepts

The Llama Stack Evaluation flow allows you to run evaluations on your GenAI application datasets or pre-registered benchmarks.

We introduce a set of APIs in Llama Stack for supporting running evaluations of LLM applications.
- `/datasetio` + `/datasets` API
- `/scoring` + `/scoring_functions` API
- `/eval` + `/benchmarks` API

This guide goes over the sets of APIs and developer experience flow of using Llama Stack to run evaluations for different use cases. Checkout our Colab notebook on working examples with evaluations [here](https://colab.research.google.com/drive/10CHyykee9j2OigaIcRv47BKG9mrNm0tJ?usp=sharing).


The Evaluation APIs are associated with a set of Resources. Please visit the Resources section in our [Core Concepts](../concepts/index.md) guide for better high-level understanding.

- **DatasetIO**: defines interface with datasets and data loaders.
  - Associated with `Dataset` resource.
- **Scoring**: evaluate outputs of the system.
  - Associated with `ScoringFunction` resource. We provide a suite of out-of-the box scoring functions and also the ability for you to add custom evaluators. These scoring functions are the core part of defining an evaluation task to output evaluation metrics.
- **Eval**: generate outputs (via Inference or Agents) and perform scoring.
  - Associated with `Benchmark` resource.


### Open-benchmark Eval

#### List of open-benchmarks Llama Stack support

Llama stack pre-registers several popular open-benchmarks to easily evaluate model perfomance via CLI.

The list of open-benchmarks we currently support:
- [MMLU-COT](https://arxiv.org/abs/2009.03300) (Measuring Massive Multitask Language Understanding): Benchmark designed to comprehensively evaluate the breadth and depth of a model's academic and professional understanding
- [GPQA-COT](https://arxiv.org/abs/2311.12022) (A Graduate-Level Google-Proof Q&A Benchmark): A challenging benchmark of 448 multiple-choice questions written by domain experts in biology, physics, and chemistry.
- [SimpleQA](https://openai.com/index/introducing-simpleqa/): Benchmark designed to access models to answer short, fact-seeking questions.
- [MMMU](https://arxiv.org/abs/2311.16502) (A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI)]: Benchmark designed to evaluate multimodal models.


You can follow this [contributing guide](../references/evals_reference/index.md#open-benchmark-contributing-guide) to add more open-benchmarks to Llama Stack

#### Run evaluation on open-benchmarks via CLI

We have built-in functionality to run the supported open-benckmarks using llama-stack-client CLI

#### Spin up Llama Stack server

Spin up llama stack server with 'open-benchmark' template
```
llama stack run llama_stack/distributions/open-benchmark/run.yaml

```

#### Run eval CLI
There are 3 necessary inputs to run a benchmark eval
- `list of benchmark_ids`: The list of benchmark ids to run evaluation on
- `model-id`: The model id to evaluate on
- `output_dir`: Path to store the evaluate results
```
llama-stack-client eval run-benchmark <benchmark_id_1> <benchmark_id_2> ... \
--model_id <model id to evaluate on> \
--output_dir <directory to store the evaluate results> \
```

You can run
```
llama-stack-client eval run-benchmark help
```
to see the description of all the flags that eval run-benchmark has


In the output log, you can find the file path that has your evaluation results. Open that file and you can see you aggregate
evaluation results over there.



#### What's Next?

- Check out our Colab notebook on working examples with running benchmark evaluations [here](https://colab.research.google.com/github/meta-llama/llama-stack/blob/main/docs/notebooks/Llama_Stack_Benchmark_Evals.ipynb#scrollTo=mxLCsP4MvFqP).
- Check out our [Building Applications - Evaluation](../building_applications/evals.md) guide for more details on how to use the Evaluation APIs to evaluate your applications.
- Check out our [Evaluation Reference](../references/evals_reference/index.md) for more details on the APIs.

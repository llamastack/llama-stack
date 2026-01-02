import type { SidebarsConfig } from "@docusaurus/plugin-content-docs";

const sidebar: SidebarsConfig = {
  apisidebar: [
    {
      type: "doc",
      id: "api-experimental/llama-stack-specification-experimental-apis",
    },
    {
      type: "category",
      label: "Agents",
      link: {
        type: "doc",
        id: "api-experimental/agents",
      },
      items: [
        {
          type: "doc",
          id: "api-experimental/list-all-agents",
          label: "List all agents.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api-experimental/create-an-agent-with-the-given-configuration",
          label: "Create an agent with the given configuration.",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api-experimental/describe-an-agent-by-its-id",
          label: "Describe an agent by its ID.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api-experimental/delete-an-agent-by-its-id-and-its-associated-sessions-and-turns",
          label: "Delete an agent by its ID and its associated sessions and turns.",
          className: "api-method delete",
        },
        {
          type: "doc",
          id: "api-experimental/create-a-new-session-for-an-agent",
          label: "Create a new session for an agent.",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api-experimental/retrieve-an-agent-session-by-its-id",
          label: "Retrieve an agent session by its ID.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api-experimental/delete-an-agent-session-by-its-id-and-its-associated-turns",
          label: "Delete an agent session by its ID and its associated turns.",
          className: "api-method delete",
        },
        {
          type: "doc",
          id: "api-experimental/create-a-new-turn-for-an-agent",
          label: "Create a new turn for an agent.",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api-experimental/retrieve-an-agent-turn-by-its-id",
          label: "Retrieve an agent turn by its ID.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api-experimental/resume-an-agent-turn-with-executed-tool-call-responses",
          label: "Resume an agent turn with executed tool call responses.",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api-experimental/retrieve-an-agent-step-by-its-id",
          label: "Retrieve an agent step by its ID.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api-experimental/list-all-session-s-of-a-given-agent",
          label: "List all session(s) of a given agent.",
          className: "api-method get",
        },
      ],
    },
    {
      type: "category",
      label: "Benchmarks",
      link: {
        type: "doc",
        id: "api-experimental/benchmarks",
      },
      items: [
        {
          type: "doc",
          id: "api-experimental/list-all-benchmarks",
          label: "List all benchmarks.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api-experimental/register-a-benchmark",
          label: "Register a benchmark.",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api-experimental/get-a-benchmark-by-its-id",
          label: "Get a benchmark by its ID.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api-experimental/unregister-a-benchmark",
          label: "Unregister a benchmark.",
          className: "api-method delete",
        },
      ],
    },
    {
      type: "category",
      label: "DatasetIO",
      link: {
        type: "doc",
        id: "api-experimental/dataset-io",
      },
      items: [
        {
          type: "doc",
          id: "api-experimental/append-rows-to-a-dataset",
          label: "Append rows to a dataset.",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api-experimental/get-a-paginated-list-of-rows-from-a-dataset",
          label: "Get a paginated list of rows from a dataset.",
          className: "api-method get",
        },
      ],
    },
    {
      type: "category",
      label: "Datasets",
      link: {
        type: "doc",
        id: "api-experimental/datasets",
      },
      items: [
        {
          type: "doc",
          id: "api-experimental/list-all-datasets",
          label: "List all datasets.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api-experimental/register-a-new-dataset",
          label: "Register a new dataset.",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api-experimental/get-a-dataset-by-its-id",
          label: "Get a dataset by its ID.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api-experimental/unregister-a-dataset-by-its-id",
          label: "Unregister a dataset by its ID.",
          className: "api-method delete",
        },
      ],
    },
    {
      type: "category",
      label: "Evaluations",
      link: {
        type: "doc",
        id: "api-experimental/eval",
      },
      items: [
        {
          type: "doc",
          id: "api-experimental/evaluate-a-list-of-rows-on-a-benchmark",
          label: "Evaluate a list of rows on a benchmark.",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api-experimental/run-an-evaluation-on-a-benchmark",
          label: "Run an evaluation on a benchmark.",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api-experimental/get-the-status-of-a-job",
          label: "Get the status of a job.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api-experimental/cancel-a-job",
          label: "Cancel a job.",
          className: "api-method delete",
        },
        {
          type: "doc",
          id: "api-experimental/get-the-result-of-a-job",
          label: "Get the result of a job.",
          className: "api-method get",
        },
      ],
    },
    {
      type: "category",
      label: "PostTraining (Coming Soon)",
      link: {
        type: "doc",
        id: "api-experimental/post-training-coming-soon",
      },
      items: [
        {
          type: "doc",
          id: "api-experimental/get-the-artifacts-of-a-training-job",
          label: "Get the artifacts of a training job.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api-experimental/cancel-a-training-job",
          label: "Cancel a training job.",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api-experimental/get-the-status-of-a-training-job",
          label: "Get the status of a training job.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api-experimental/get-all-training-jobs",
          label: "Get all training jobs.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api-experimental/run-preference-optimization-of-a-model",
          label: "Run preference optimization of a model.",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api-experimental/run-supervised-fine-tuning-of-a-model",
          label: "Run supervised fine-tuning of a model.",
          className: "api-method post",
        },
      ],
    },
    {
      type: "category",
      label: "Inference",
      items: [
        {
          type: "doc",
          id: "api-experimental/rerank-a-list-of-documents-based-on-their-relevance-to-a-query",
          label: "Rerank a list of documents based on their relevance to a query.",
          className: "api-method post",
        },
      ],
    },
  ],
};

export default sidebar.apisidebar;

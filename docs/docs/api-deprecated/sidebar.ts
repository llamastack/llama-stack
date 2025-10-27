import type { SidebarsConfig } from "@docusaurus/plugin-content-docs";

const sidebar: SidebarsConfig = {
  apisidebar: [
    {
      type: "doc",
      id: "api-deprecated/llama-stack-specification-deprecated-apis",
    },
    {
      type: "category",
      label: "Agents",
      link: {
        type: "doc",
        id: "api-deprecated/agents",
      },
      items: [
        {
          type: "doc",
          id: "api-deprecated/list-all-agents",
          label: "List all agents.",
          className: "menu__list-item--deprecated api-method get",
        },
        {
          type: "doc",
          id: "api-deprecated/create-an-agent-with-the-given-configuration",
          label: "Create an agent with the given configuration.",
          className: "menu__list-item--deprecated api-method post",
        },
        {
          type: "doc",
          id: "api-deprecated/describe-an-agent-by-its-id",
          label: "Describe an agent by its ID.",
          className: "menu__list-item--deprecated api-method get",
        },
        {
          type: "doc",
          id: "api-deprecated/delete-an-agent-by-its-id-and-its-associated-sessions-and-turns",
          label: "Delete an agent by its ID and its associated sessions and turns.",
          className: "menu__list-item--deprecated api-method delete",
        },
        {
          type: "doc",
          id: "api-deprecated/create-a-new-session-for-an-agent",
          label: "Create a new session for an agent.",
          className: "menu__list-item--deprecated api-method post",
        },
        {
          type: "doc",
          id: "api-deprecated/retrieve-an-agent-session-by-its-id",
          label: "Retrieve an agent session by its ID.",
          className: "menu__list-item--deprecated api-method get",
        },
        {
          type: "doc",
          id: "api-deprecated/delete-an-agent-session-by-its-id-and-its-associated-turns",
          label: "Delete an agent session by its ID and its associated turns.",
          className: "menu__list-item--deprecated api-method delete",
        },
        {
          type: "doc",
          id: "api-deprecated/create-a-new-turn-for-an-agent",
          label: "Create a new turn for an agent.",
          className: "menu__list-item--deprecated api-method post",
        },
        {
          type: "doc",
          id: "api-deprecated/retrieve-an-agent-turn-by-its-id",
          label: "Retrieve an agent turn by its ID.",
          className: "menu__list-item--deprecated api-method get",
        },
        {
          type: "doc",
          id: "api-deprecated/resume-an-agent-turn-with-executed-tool-call-responses",
          label: "Resume an agent turn with executed tool call responses.",
          className: "menu__list-item--deprecated api-method post",
        },
        {
          type: "doc",
          id: "api-deprecated/retrieve-an-agent-step-by-its-id",
          label: "Retrieve an agent step by its ID.",
          className: "menu__list-item--deprecated api-method get",
        },
        {
          type: "doc",
          id: "api-deprecated/list-all-session-s-of-a-given-agent",
          label: "List all session(s) of a given agent.",
          className: "menu__list-item--deprecated api-method get",
        },
        {
          type: "doc",
          id: "api-deprecated/list-all-responses",
          label: "List all responses.",
          className: "menu__list-item--deprecated api-method get",
        },
        {
          type: "doc",
          id: "api-deprecated/create-a-model-response",
          label: "Create a model response.",
          className: "menu__list-item--deprecated api-method post",
        },
        {
          type: "doc",
          id: "api-deprecated/get-a-model-response",
          label: "Get a model response.",
          className: "menu__list-item--deprecated api-method get",
        },
        {
          type: "doc",
          id: "api-deprecated/delete-a-response",
          label: "Delete a response.",
          className: "menu__list-item--deprecated api-method delete",
        },
        {
          type: "doc",
          id: "api-deprecated/list-input-items",
          label: "List input items.",
          className: "menu__list-item--deprecated api-method get",
        },
      ],
    },
    {
      type: "category",
      label: "Benchmarks",
      link: {
        type: "doc",
        id: "api-deprecated/benchmarks",
      },
      items: [
        {
          type: "doc",
          id: "api-deprecated/list-all-benchmarks",
          label: "List all benchmarks.",
          className: "menu__list-item--deprecated api-method get",
        },
        {
          type: "doc",
          id: "api-deprecated/register-a-benchmark",
          label: "Register a benchmark.",
          className: "menu__list-item--deprecated api-method post",
        },
        {
          type: "doc",
          id: "api-deprecated/get-a-benchmark-by-its-id",
          label: "Get a benchmark by its ID.",
          className: "menu__list-item--deprecated api-method get",
        },
        {
          type: "doc",
          id: "api-deprecated/unregister-a-benchmark",
          label: "Unregister a benchmark.",
          className: "menu__list-item--deprecated api-method delete",
        },
      ],
    },
    {
      type: "category",
      label: "DatasetIO",
      link: {
        type: "doc",
        id: "api-deprecated/dataset-io",
      },
      items: [
        {
          type: "doc",
          id: "api-deprecated/append-rows-to-a-dataset",
          label: "Append rows to a dataset.",
          className: "menu__list-item--deprecated api-method post",
        },
        {
          type: "doc",
          id: "api-deprecated/get-a-paginated-list-of-rows-from-a-dataset",
          label: "Get a paginated list of rows from a dataset.",
          className: "menu__list-item--deprecated api-method get",
        },
      ],
    },
    {
      type: "category",
      label: "Datasets",
      link: {
        type: "doc",
        id: "api-deprecated/datasets",
      },
      items: [
        {
          type: "doc",
          id: "api-deprecated/list-all-datasets",
          label: "List all datasets.",
          className: "menu__list-item--deprecated api-method get",
        },
        {
          type: "doc",
          id: "api-deprecated/register-a-new-dataset",
          label: "Register a new dataset.",
          className: "menu__list-item--deprecated api-method post",
        },
        {
          type: "doc",
          id: "api-deprecated/get-a-dataset-by-its-id",
          label: "Get a dataset by its ID.",
          className: "menu__list-item--deprecated api-method get",
        },
        {
          type: "doc",
          id: "api-deprecated/unregister-a-dataset-by-its-id",
          label: "Unregister a dataset by its ID.",
          className: "menu__list-item--deprecated api-method delete",
        },
      ],
    },
    {
      type: "category",
      label: "Evaluations",
      link: {
        type: "doc",
        id: "api-deprecated/eval",
      },
      items: [
        {
          type: "doc",
          id: "api-deprecated/evaluate-a-list-of-rows-on-a-benchmark",
          label: "Evaluate a list of rows on a benchmark.",
          className: "menu__list-item--deprecated api-method post",
        },
        {
          type: "doc",
          id: "api-deprecated/run-an-evaluation-on-a-benchmark",
          label: "Run an evaluation on a benchmark.",
          className: "menu__list-item--deprecated api-method post",
        },
        {
          type: "doc",
          id: "api-deprecated/get-the-status-of-a-job",
          label: "Get the status of a job.",
          className: "menu__list-item--deprecated api-method get",
        },
        {
          type: "doc",
          id: "api-deprecated/cancel-a-job",
          label: "Cancel a job.",
          className: "menu__list-item--deprecated api-method delete",
        },
        {
          type: "doc",
          id: "api-deprecated/get-the-result-of-a-job",
          label: "Get the result of a job.",
          className: "menu__list-item--deprecated api-method get",
        },
      ],
    },
    {
      type: "category",
      label: "Files",
      link: {
        type: "doc",
        id: "api-deprecated/files",
      },
      items: [
        {
          type: "doc",
          id: "api-deprecated/list-files",
          label: "List files.",
          className: "menu__list-item--deprecated api-method get",
        },
        {
          type: "doc",
          id: "api-deprecated/upload-file",
          label: "Upload file.",
          className: "menu__list-item--deprecated api-method post",
        },
        {
          type: "doc",
          id: "api-deprecated/retrieve-file",
          label: "Retrieve file.",
          className: "menu__list-item--deprecated api-method get",
        },
        {
          type: "doc",
          id: "api-deprecated/delete-file",
          label: "Delete file.",
          className: "menu__list-item--deprecated api-method delete",
        },
        {
          type: "doc",
          id: "api-deprecated/retrieve-file-content",
          label: "Retrieve file content.",
          className: "menu__list-item--deprecated api-method get",
        },
      ],
    },
    {
      type: "category",
      label: "Inference",
      link: {
        type: "doc",
        id: "api-deprecated/inference",
      },
      items: [
        {
          type: "doc",
          id: "api-deprecated/list-chat-completions",
          label: "List chat completions.",
          className: "menu__list-item--deprecated api-method get",
        },
        {
          type: "doc",
          id: "api-deprecated/create-chat-completions",
          label: "Create chat completions.",
          className: "menu__list-item--deprecated api-method post",
        },
        {
          type: "doc",
          id: "api-deprecated/get-chat-completion",
          label: "Get chat completion.",
          className: "menu__list-item--deprecated api-method get",
        },
        {
          type: "doc",
          id: "api-deprecated/create-completion",
          label: "Create completion.",
          className: "menu__list-item--deprecated api-method post",
        },
        {
          type: "doc",
          id: "api-deprecated/create-embeddings",
          label: "Create embeddings.",
          className: "menu__list-item--deprecated api-method post",
        },
      ],
    },
    {
      type: "category",
      label: "Models",
      link: {
        type: "doc",
        id: "api-deprecated/models",
      },
      items: [
        {
          type: "doc",
          id: "api-deprecated/list-models-using-the-open-ai-api",
          label: "List models using the OpenAI API.",
          className: "menu__list-item--deprecated api-method get",
        },
      ],
    },
    {
      type: "category",
      label: "PostTraining (Coming Soon)",
      link: {
        type: "doc",
        id: "api-deprecated/post-training-coming-soon",
      },
      items: [
        {
          type: "doc",
          id: "api-deprecated/get-the-artifacts-of-a-training-job",
          label: "Get the artifacts of a training job.",
          className: "menu__list-item--deprecated api-method get",
        },
        {
          type: "doc",
          id: "api-deprecated/cancel-a-training-job",
          label: "Cancel a training job.",
          className: "menu__list-item--deprecated api-method post",
        },
        {
          type: "doc",
          id: "api-deprecated/get-the-status-of-a-training-job",
          label: "Get the status of a training job.",
          className: "menu__list-item--deprecated api-method get",
        },
        {
          type: "doc",
          id: "api-deprecated/get-all-training-jobs",
          label: "Get all training jobs.",
          className: "menu__list-item--deprecated api-method get",
        },
        {
          type: "doc",
          id: "api-deprecated/run-preference-optimization-of-a-model",
          label: "Run preference optimization of a model.",
          className: "menu__list-item--deprecated api-method post",
        },
        {
          type: "doc",
          id: "api-deprecated/run-supervised-fine-tuning-of-a-model",
          label: "Run supervised fine-tuning of a model.",
          className: "menu__list-item--deprecated api-method post",
        },
      ],
    },
    {
      type: "category",
      label: "Safety",
      link: {
        type: "doc",
        id: "api-deprecated/safety",
      },
      items: [
        {
          type: "doc",
          id: "api-deprecated/create-moderation",
          label: "Create moderation.",
          className: "menu__list-item--deprecated api-method post",
        },
      ],
    },
    {
      type: "category",
      label: "VectorIO",
      link: {
        type: "doc",
        id: "api-deprecated/vector-io",
      },
      items: [
        {
          type: "doc",
          id: "api-deprecated/returns-a-list-of-vector-stores",
          label: "Returns a list of vector stores.",
          className: "menu__list-item--deprecated api-method get",
        },
        {
          type: "doc",
          id: "api-deprecated/creates-a-vector-store",
          label: "Creates a vector store.",
          className: "menu__list-item--deprecated api-method post",
        },
        {
          type: "doc",
          id: "api-deprecated/retrieves-a-vector-store",
          label: "Retrieves a vector store.",
          className: "menu__list-item--deprecated api-method get",
        },
        {
          type: "doc",
          id: "api-deprecated/updates-a-vector-store",
          label: "Updates a vector store.",
          className: "menu__list-item--deprecated api-method post",
        },
        {
          type: "doc",
          id: "api-deprecated/delete-a-vector-store",
          label: "Delete a vector store.",
          className: "menu__list-item--deprecated api-method delete",
        },
        {
          type: "doc",
          id: "api-deprecated/create-a-vector-store-file-batch",
          label: "Create a vector store file batch.",
          className: "menu__list-item--deprecated api-method post",
        },
        {
          type: "doc",
          id: "api-deprecated/retrieve-a-vector-store-file-batch",
          label: "Retrieve a vector store file batch.",
          className: "menu__list-item--deprecated api-method get",
        },
        {
          type: "doc",
          id: "api-deprecated/cancels-a-vector-store-file-batch",
          label: "Cancels a vector store file batch.",
          className: "menu__list-item--deprecated api-method post",
        },
        {
          type: "doc",
          id: "api-deprecated/returns-a-list-of-vector-store-files-in-a-batch",
          label: "Returns a list of vector store files in a batch.",
          className: "menu__list-item--deprecated api-method get",
        },
        {
          type: "doc",
          id: "api-deprecated/list-files-in-a-vector-store",
          label: "List files in a vector store.",
          className: "menu__list-item--deprecated api-method get",
        },
        {
          type: "doc",
          id: "api-deprecated/attach-a-file-to-a-vector-store",
          label: "Attach a file to a vector store.",
          className: "menu__list-item--deprecated api-method post",
        },
        {
          type: "doc",
          id: "api-deprecated/retrieves-a-vector-store-file",
          label: "Retrieves a vector store file.",
          className: "menu__list-item--deprecated api-method get",
        },
        {
          type: "doc",
          id: "api-deprecated/updates-a-vector-store-file",
          label: "Updates a vector store file.",
          className: "menu__list-item--deprecated api-method post",
        },
        {
          type: "doc",
          id: "api-deprecated/delete-a-vector-store-file",
          label: "Delete a vector store file.",
          className: "menu__list-item--deprecated api-method delete",
        },
        {
          type: "doc",
          id: "api-deprecated/retrieves-the-contents-of-a-vector-store-file",
          label: "Retrieves the contents of a vector store file.",
          className: "menu__list-item--deprecated api-method get",
        },
        {
          type: "doc",
          id: "api-deprecated/search-for-chunks-in-a-vector-store",
          label: "Search for chunks in a vector store.",
          className: "menu__list-item--deprecated api-method post",
        },
      ],
    },
  ],
};

export default sidebar.apisidebar;

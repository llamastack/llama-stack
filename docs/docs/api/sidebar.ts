import type { SidebarsConfig } from "@docusaurus/plugin-content-docs";

const sidebar: SidebarsConfig = {
  apisidebar: [
    {
      type: "doc",
      id: "api/llama-stack-specification",
    },
    {
      type: "category",
      label: "Agents",
      link: {
        type: "doc",
        id: "api/agents",
      },
      items: [
        {
          type: "doc",
          id: "api/list-all-responses",
          label: "List all responses.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api/create-a-model-response",
          label: "Create a model response.",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api/get-a-model-response",
          label: "Get a model response.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api/delete-a-response",
          label: "Delete a response.",
          className: "api-method delete",
        },
        {
          type: "doc",
          id: "api/list-input-items",
          label: "List input items.",
          className: "api-method get",
        },
      ],
    },
    {
      type: "category",
      label: "Conversations",
      link: {
        type: "doc",
        id: "api/conversations",
      },
      items: [
        {
          type: "doc",
          id: "api/create-a-conversation",
          label: "Create a conversation.",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api/retrieve-a-conversation",
          label: "Retrieve a conversation.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api/update-a-conversation",
          label: "Update a conversation.",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api/delete-a-conversation",
          label: "Delete a conversation.",
          className: "api-method delete",
        },
        {
          type: "doc",
          id: "api/list-items",
          label: "List items.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api/create-items",
          label: "Create items.",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api/retrieve-an-item",
          label: "Retrieve an item.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api/delete-an-item",
          label: "Delete an item.",
          className: "api-method delete",
        },
      ],
    },
    {
      type: "category",
      label: "Files",
      link: {
        type: "doc",
        id: "api/files",
      },
      items: [
        {
          type: "doc",
          id: "api/list-files",
          label: "List files.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api/upload-file",
          label: "Upload file.",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api/retrieve-file",
          label: "Retrieve file.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api/delete-file",
          label: "Delete file.",
          className: "api-method delete",
        },
        {
          type: "doc",
          id: "api/retrieve-file-content",
          label: "Retrieve file content.",
          className: "api-method get",
        },
      ],
    },
    {
      type: "category",
      label: "Inference",
      link: {
        type: "doc",
        id: "api/inference",
      },
      items: [
        {
          type: "doc",
          id: "api/list-chat-completions",
          label: "List chat completions.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api/create-chat-completions",
          label: "Create chat completions.",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api/get-chat-completion",
          label: "Get chat completion.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api/create-completion",
          label: "Create completion.",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api/create-embeddings",
          label: "Create embeddings.",
          className: "api-method post",
        },
      ],
    },
    {
      type: "category",
      label: "Inspect",
      link: {
        type: "doc",
        id: "api/inspect",
      },
      items: [
        {
          type: "doc",
          id: "api/get-health-status",
          label: "Get health status.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api/list-routes",
          label: "List routes.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api/get-version",
          label: "Get version.",
          className: "api-method get",
        },
      ],
    },
    {
      type: "category",
      label: "Models",
      link: {
        type: "doc",
        id: "api/models",
      },
      items: [
        {
          type: "doc",
          id: "api/list-all-models",
          label: "List all models.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api/register-model",
          label: "Register model.",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api/get-model",
          label: "Get model.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api/unregister-model",
          label: "Unregister model.",
          className: "api-method delete",
        },
      ],
    },
    {
      type: "category",
      label: "Prompts",
      link: {
        type: "doc",
        id: "api/prompts",
      },
      items: [
        {
          type: "doc",
          id: "api/list-all-prompts",
          label: "List all prompts.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api/create-prompt",
          label: "Create prompt.",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api/get-prompt",
          label: "Get prompt.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api/update-prompt",
          label: "Update prompt.",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api/delete-prompt",
          label: "Delete prompt.",
          className: "api-method delete",
        },
        {
          type: "doc",
          id: "api/set-prompt-version",
          label: "Set prompt version.",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api/list-prompt-versions",
          label: "List prompt versions.",
          className: "api-method get",
        },
      ],
    },
    {
      type: "category",
      label: "Providers",
      link: {
        type: "doc",
        id: "api/providers",
      },
      items: [
        {
          type: "doc",
          id: "api/list-providers",
          label: "List providers.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api/get-provider",
          label: "Get provider.",
          className: "api-method get",
        },
      ],
    },
    {
      type: "category",
      label: "Safety",
      link: {
        type: "doc",
        id: "api/safety",
      },
      items: [
        {
          type: "doc",
          id: "api/create-moderation",
          label: "Create moderation.",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api/run-shield",
          label: "Run shield.",
          className: "api-method post",
        },
      ],
    },
    {
      type: "category",
      label: "Scoring",
      link: {
        type: "doc",
        id: "api/scoring",
      },
      items: [
        {
          type: "doc",
          id: "api/score-a-list-of-rows",
          label: "Score a list of rows.",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api/score-a-batch-of-rows",
          label: "Score a batch of rows.",
          className: "api-method post",
        },
      ],
    },
    {
      type: "category",
      label: "ScoringFunctions",
      link: {
        type: "doc",
        id: "api/scoring-functions",
      },
      items: [
        {
          type: "doc",
          id: "api/list-all-scoring-functions",
          label: "List all scoring functions.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api/register-a-scoring-function",
          label: "Register a scoring function.",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api/get-a-scoring-function-by-its-id",
          label: "Get a scoring function by its ID.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api/unregister-a-scoring-function",
          label: "Unregister a scoring function.",
          className: "api-method delete",
        },
      ],
    },
    {
      type: "category",
      label: "Shields",
      link: {
        type: "doc",
        id: "api/shields",
      },
      items: [
        {
          type: "doc",
          id: "api/list-all-shields",
          label: "List all shields.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api/register-a-shield",
          label: "Register a shield.",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api/get-a-shield-by-its-identifier",
          label: "Get a shield by its identifier.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api/unregister-a-shield",
          label: "Unregister a shield.",
          className: "api-method delete",
        },
      ],
    },
    {
      type: "category",
      label: "SyntheticDataGeneration (Coming Soon)",
      link: {
        type: "doc",
        id: "api/synthetic-data-generation-coming-soon",
      },
      items: [
        {
          type: "doc",
          id: "api/generate-synthetic-data-based-on-input-dialogs-and-apply-filtering",
          label: "Generate synthetic data based on input dialogs and apply filtering.",
          className: "api-method post",
        },
      ],
    },
    {
      type: "category",
      label: "ToolGroups",
      link: {
        type: "doc",
        id: "api/tool-groups",
      },
      items: [
        {
          type: "doc",
          id: "api/list-tool-groups-with-optional-provider",
          label: "List tool groups with optional provider.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api/register-a-tool-group",
          label: "Register a tool group.",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api/get-a-tool-group-by-its-id",
          label: "Get a tool group by its ID.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api/unregister-a-tool-group",
          label: "Unregister a tool group.",
          className: "api-method delete",
        },
        {
          type: "doc",
          id: "api/list-tools-with-optional-tool-group",
          label: "List tools with optional tool group.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api/get-a-tool-by-its-name",
          label: "Get a tool by its name.",
          className: "api-method get",
        },
      ],
    },
    {
      type: "category",
      label: "ToolRuntime",
      link: {
        type: "doc",
        id: "api/tool-runtime",
      },
      items: [
        {
          type: "doc",
          id: "api/run-a-tool-with-the-given-arguments",
          label: "Run a tool with the given arguments.",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api/list-all-tools-in-the-runtime",
          label: "List all tools in the runtime.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api/index-documents-so-they-can-be-used-by-the-rag-system",
          label: "Index documents so they can be used by the RAG system.",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api/query-the-rag-system-for-context-typically-invoked-by-the-agent",
          label: "Query the RAG system for context; typically invoked by the agent.",
          className: "api-method post",
        },
      ],
    },
    {
      type: "category",
      label: "VectorIO",
      link: {
        type: "doc",
        id: "api/vector-io",
      },
      items: [
        {
          type: "doc",
          id: "api/insert-chunks-into-a-vector-database",
          label: "Insert chunks into a vector database.",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api/query-chunks-from-a-vector-database",
          label: "Query chunks from a vector database.",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api/returns-a-list-of-vector-stores",
          label: "Returns a list of vector stores.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api/creates-a-vector-store",
          label: "Creates a vector store.",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api/retrieves-a-vector-store",
          label: "Retrieves a vector store.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api/updates-a-vector-store",
          label: "Updates a vector store.",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api/delete-a-vector-store",
          label: "Delete a vector store.",
          className: "api-method delete",
        },
        {
          type: "doc",
          id: "api/create-a-vector-store-file-batch",
          label: "Create a vector store file batch.",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api/retrieve-a-vector-store-file-batch",
          label: "Retrieve a vector store file batch.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api/cancels-a-vector-store-file-batch",
          label: "Cancels a vector store file batch.",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api/returns-a-list-of-vector-store-files-in-a-batch",
          label: "Returns a list of vector store files in a batch.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api/list-files-in-a-vector-store",
          label: "List files in a vector store.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api/attach-a-file-to-a-vector-store",
          label: "Attach a file to a vector store.",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api/retrieves-a-vector-store-file",
          label: "Retrieves a vector store file.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api/updates-a-vector-store-file",
          label: "Updates a vector store file.",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api/delete-a-vector-store-file",
          label: "Delete a vector store file.",
          className: "api-method delete",
        },
        {
          type: "doc",
          id: "api/retrieves-the-contents-of-a-vector-store-file",
          label: "Retrieves the contents of a vector store file.",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api/search-for-chunks-in-a-vector-store",
          label: "Search for chunks in a vector store.",
          className: "api-method post",
        },
      ],
    },
  ],
};

export default sidebar.apisidebar;

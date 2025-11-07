import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    'index',
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: true,
      items: [
        'getting_started/quickstart',
        'getting_started/detailed_tutorial',
        'getting_started/libraries',
      ],
    },
    {
      type: 'category',
      label: 'Concepts',
      collapsed: true,
      items: [
        'concepts/index',
        'concepts/architecture',
        {
          type: 'category',
          label: 'APIs',
          collapsed: true,
          items: [
            'concepts/apis/index',
            'concepts/apis/api_providers',
            'concepts/apis/external',
            'concepts/apis/api_leveling',
          ],
        },
        'concepts/distributions',
        'concepts/resources',
      ],
    },
    {
      type: 'category',
      label: 'Distributions',
      collapsed: true,
      items: [
        'distributions/index',
        'distributions/list_of_distributions',
        'distributions/building_distro',
        'distributions/customizing_run_yaml',
        'distributions/importing_as_library',
        'distributions/configuration',
        'distributions/starting_llama_stack_server',
        'distributions/llama_stack_ui',
        {
          type: 'category',
          label: 'Self-Hosted Distributions',
          collapsed: true,
          items: [
            'distributions/self_hosted_distro/starter',
            'distributions/self_hosted_distro/dell',
            'distributions/self_hosted_distro/dell-tgi',
            'distributions/self_hosted_distro/meta-reference-gpu',
            'distributions/self_hosted_distro/nvidia',
            'distributions/self_hosted_distro/passthrough',
          ],
        },
        {
          type: 'category',
          label: 'Remote-Hosted Distributions',
          collapsed: true,
          items: [
            'distributions/remote_hosted_distro/index',
            'distributions/remote_hosted_distro/watsonx',
          ],
        },
        {
          type: 'category',
          label: 'On-Device Distributions',
          collapsed: true,
          items: [
            'distributions/ondevice_distro/ios_sdk',
            'distributions/ondevice_distro/android_sdk',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Providers',
      collapsed: true,
      items: [
        'providers/index',
        {
          type: 'category',
          label: 'Inference',
          collapsed: true,
          items: [
            'providers/inference/index',
            'providers/inference/inline_meta-reference',
            'providers/inference/inline_sentence-transformers',
            'providers/inference/remote_anthropic',
            'providers/inference/remote_azure',
            'providers/inference/remote_bedrock',
            'providers/inference/remote_cerebras',
            'providers/inference/remote_databricks',
            'providers/inference/remote_fireworks',
            'providers/inference/remote_gemini',
            'providers/inference/remote_groq',
            'providers/inference/remote_hf_endpoint',
            'providers/inference/remote_hf_serverless',
            'providers/inference/remote_llama-openai-compat',
            'providers/inference/remote_nvidia',
            'providers/inference/remote_ollama',
            'providers/inference/remote_openai',
            'providers/inference/remote_passthrough',
            'providers/inference/remote_runpod',
            'providers/inference/remote_sambanova',
            'providers/inference/remote_sambanova-openai-compat',
            'providers/inference/remote_tgi',
            'providers/inference/remote_together',
            'providers/inference/remote_vertexai',
            'providers/inference/remote_vllm',
            'providers/inference/remote_watsonx'
          ],
        },
        {
          type: 'category',
          label: 'Safety',
          collapsed: true,
          items: [
            'providers/safety/index',
            'providers/safety/inline_code-scanner',
            'providers/safety/inline_llama-guard',
            'providers/safety/inline_prompt-guard',
            'providers/safety/remote_bedrock',
            'providers/safety/remote_nvidia',
            'providers/safety/remote_sambanova'
          ],
        },
        {
          type: 'category',
          label: 'Vector IO',
          collapsed: true,
          items: [
            'providers/vector_io/index',
            'providers/vector_io/inline_chromadb',
            'providers/vector_io/inline_faiss',
            'providers/vector_io/inline_meta-reference',
            'providers/vector_io/inline_milvus',
            'providers/vector_io/inline_qdrant',
            'providers/vector_io/inline_sqlite-vec',
            'providers/vector_io/remote_chromadb',
            'providers/vector_io/remote_milvus',
            'providers/vector_io/remote_pgvector',
            'providers/vector_io/remote_qdrant',
            'providers/vector_io/remote_weaviate'
          ],
        },
        {
          type: 'category',
          label: 'Tool Runtime',
          collapsed: true,
          items: [
            'providers/tool_runtime/index',
            'providers/tool_runtime/inline_rag-runtime',
            'providers/tool_runtime/remote_bing-search',
            'providers/tool_runtime/remote_brave-search',
            'providers/tool_runtime/remote_model-context-protocol',
            'providers/tool_runtime/remote_tavily-search',
            'providers/tool_runtime/remote_wolfram-alpha'
          ],
        },
        {
          type: 'category',
          label: 'Agents',
          collapsed: true,
          items: [
            'providers/agents/index',
            'providers/agents/inline_meta-reference'
          ],
        },
        {
          type: 'category',
          label: 'Post Training',
          collapsed: true,
          items: [
            'providers/post_training/index',
            'providers/post_training/inline_huggingface',
            'providers/post_training/inline_huggingface-cpu',
            'providers/post_training/inline_huggingface-gpu',
            'providers/post_training/inline_torchtune',
            'providers/post_training/inline_torchtune-cpu',
            'providers/post_training/inline_torchtune-gpu',
            'providers/post_training/remote_nvidia'
          ],
        },
        {
          type: 'category',
          label: 'DatasetIO',
          collapsed: true,
          items: [
            'providers/datasetio/index',
            'providers/datasetio/inline_localfs',
            'providers/datasetio/remote_huggingface',
            'providers/datasetio/remote_nvidia'
          ],
        },
        {
          type: 'category',
          label: 'Scoring',
          collapsed: true,
          items: [
            'providers/scoring/index',
            'providers/scoring/inline_basic',
            'providers/scoring/inline_braintrust',
            'providers/scoring/inline_llm-as-judge'
          ],
        },
        {
          type: 'category',
          label: 'Files',
          collapsed: true,
          items: [
            'providers/files/index',
            'providers/files/inline_localfs',
            'providers/files/remote_s3'
          ],
        },
        {
          type: 'category',
          label: 'Eval',
          collapsed: true,
          items: [
            'providers/eval/index',
            'providers/eval/inline_meta-reference',
            'providers/eval/remote_nvidia'
          ],
        },
        {
          type: 'category',
          label: 'Batches',
          collapsed: true,
          items: [
            'providers/batches/index',
            'providers/batches/inline_reference'
          ],
        },
        {
          type: 'category',
          label: 'External Providers',
          collapsed: true,
          items: [
            'providers/external/index',
            'providers/external/external-providers-guide',
            'providers/external/external-providers-list'
          ],
        },
        'providers/openai'
      ],
    },
    {
      type: 'category',
      label: 'Building Applications',
      collapsed: true,
      items: [
        'building_applications/index',
        'building_applications/rag',
        'building_applications/agent',
        'building_applications/agent_execution_loop',
        'building_applications/responses_vs_agents',
        'building_applications/tools',
        'building_applications/evals',
        'building_applications/telemetry',
        'building_applications/safety',
        'building_applications/playground',
      ],
    },
    {
      type: 'category',
      label: 'Advanced APIs',
      collapsed: true,
      items: [
        'advanced_apis/post_training',
        'advanced_apis/evaluation',
        'advanced_apis/scoring',
      ],
    },
    {
      type: 'category',
      label: 'Deploying',
      collapsed: true,
      items: [
        'deploying/index',
        'deploying/kubernetes_deployment',
        'deploying/aws_eks_deployment',
      ],
    },
    {
      type: 'category',
      label: 'Contributing',
      collapsed: true,
      items: [
        'contributing/index',
        'contributing/new_api_provider',
        'contributing/new_vector_database',
        'contributing/testing/record-replay',
      ],
    },
    {
      type: 'category',
      label: 'References',
      collapsed: true,
      items: [
        'references/index',
        'references/llama_cli_reference/index',
        'references/llama_stack_client_cli_reference',
        'references/python_sdk_reference/index',
        'references/evals_reference/index',
      ],
    },
  ],

  // API Reference sidebars - use plugin-generated sidebars
  stableApiSidebar: require('./docs/api/sidebar.ts').default,
  experimentalApiSidebar: require('./docs/api-experimental/sidebar.ts').default,
  deprecatedApiSidebar: require('./docs/api-deprecated/sidebar.ts').default,
};

export default sidebars;

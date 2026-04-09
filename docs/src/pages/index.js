import React, {useState} from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import {useColorMode} from '@docusaurus/theme-common';
import {Highlight, themes} from 'prism-react-renderer';
import InstallBlock from '../components/InstallBlock';
import styles from './index.module.css';

const LANG_TO_PRISM = {
  'Python': 'python',
  'curl': 'bash',
  'Node.js': 'javascript',
  'Go': 'go',
  'TypeScript': 'typescript',
};

const SDK_EXAMPLES = {
  openai: {
    label: 'OpenAI SDK',
    endpoint: '/v1/responses',
    languages: [
      {
        lang: 'Python',
        code: `from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")
response = client.responses.create(
    model="llama-3.3-70b",
    input="Summarize this repository",
    tools=[{"type": "web_search"}],
)`,
      },
      {
        lang: 'curl',
        code: `curl http://localhost:8321/v1/responses \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "llama-3.3-70b",
    "input": "Summarize this repository",
    "tools": [{"type": "web_search"}]
  }'`,
      },
      {
        lang: 'Node.js',
        code: `import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8321/v1",
  apiKey: "fake",
});

const response = await client.responses.create({
  model: "llama-3.3-70b",
  input: "Summarize this repository",
  tools: [{ type: "web_search" }],
});`,
      },
      {
        lang: 'Go',
        code: `client := openai.NewClient(
  option.WithBaseURL("http://localhost:8321/v1"),
  option.WithAPIKey("fake"),
)

response, err := client.Responses.New(
  context.TODO(),
  openai.ResponseNewParams{
    Model: "llama-3.3-70b",
    Input: "Summarize this repository",
    Tools: []ResponseToolUnionParam{
      openai.WebSearchTool(),
    },
  },
)`,
      },
    ],
  },
  anthropic: {
    label: 'Anthropic SDK',
    endpoint: '/v1/messages',
    languages: [
      {
        lang: 'Python',
        code: `from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:8321/v1",
    api_key="fake",
)
message = client.messages.create(
    model="llama-3.3-70b",
    max_tokens=1024,
    messages=[
        {"role": "user",
         "content": "Summarize this repository"}
    ],
)`,
      },
      {
        lang: 'TypeScript',
        code: `import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic({
  baseURL: "http://localhost:8321/v1",
  apiKey: "fake",
});

const message = await client.messages.create({
  model: "llama-3.3-70b",
  max_tokens: 1024,
  messages: [{ role: "user", content: "Summarize this repository" }],
});`,
      },
    ],
  },
};

const API_SURFACE = [
  { category: 'Inference', endpoints: [
    { label: 'Chat Completions', path: '/v1/chat/completions' },
    { label: 'Responses', path: '/v1/responses' },
    { label: 'Embeddings', path: '/v1/embeddings' },
    { label: 'Models', path: '/v1/models' },
    { label: 'Messages', path: '/v1/messages', note: 'Anthropic' },
  ]},
  { category: 'Data', endpoints: [
    { label: 'Vector Stores', path: '/v1/vector_stores' },
    { label: 'Files', path: '/v1/files' },
    { label: 'Batches', path: '/v1/batches' },
  ]},
  { category: 'Safety & Tools', endpoints: [
    { label: 'Moderations', path: '/v1/moderations' },
    { label: 'Tools', path: '/v1/tools' },
    { label: 'Connectors', path: '/v1/connectors' },
  ]},
];

const PROVIDERS = [
  { name: 'Ollama', href: '/docs/providers/inference/remote_ollama' },
  { name: 'vLLM', href: '/docs/providers/inference/remote_llama-openai-compat' },
  { name: 'OpenAI', href: '/docs/providers/inference/remote_openai' },
  { name: 'Anthropic', href: '/docs/providers/inference/remote_anthropic' },
  { name: 'AWS Bedrock', href: '/docs/providers/inference/remote_bedrock' },
  { name: 'Azure OpenAI', href: '/docs/providers/inference/remote_azure' },
  { name: 'Gemini', href: '/docs/providers/inference/remote_gemini' },
  { name: 'Together AI', href: '/docs/providers/inference/remote_together' },
  { name: 'Fireworks', href: '/docs/providers/inference/remote_fireworks' },
  { name: 'PGVector', href: '/docs/providers/vector_io/remote_pgvector' },
  { name: 'Qdrant', href: '/docs/providers/vector_io/remote_qdrant' },
  { name: 'ChromaDB', href: '/docs/providers/vector_io/remote_chromadb' },
  { name: 'Milvus', href: '/docs/providers/vector_io/remote_milvus' },
  { name: 'Weaviate', href: '/docs/providers/vector_io/remote_weaviate' },
];

/* Custom Tidal theme for the landing page code block */
const tidalDark = {
  plain: { color: '#bcc5d0', backgroundColor: 'transparent' },
  styles: [
    { types: ['comment', 'prolog'], style: { color: '#546678' } },
    { types: ['keyword', 'builtin'], style: { color: '#7eb8d4' } },
    { types: ['string', 'attr-value', 'char'], style: { color: '#d4a55a' } },
    { types: ['function'], style: { color: '#2dbdc2' } },
    { types: ['class-name'], style: { color: '#c9a84c' } },
    { types: ['number', 'boolean'], style: { color: '#c9a84c' } },
    { types: ['operator'], style: { color: '#45cace' } },
    { types: ['punctuation'], style: { color: '#8b98a8' } },
    { types: ['property', 'constant'], style: { color: '#d4856a' } },
    { types: ['variable'], style: { color: '#bcc5d0' } },
  ],
};

const tidalLight = {
  plain: { color: '#2d3748', backgroundColor: 'transparent' },
  styles: [
    { types: ['comment', 'prolog'], style: { color: '#8393a7' } },
    { types: ['keyword', 'builtin'], style: { color: '#3d6b8e' } },
    { types: ['string', 'attr-value', 'char'], style: { color: '#8a6e2f' } },
    { types: ['function'], style: { color: '#0b6165' } },
    { types: ['class-name'], style: { color: '#7c6322' } },
    { types: ['number', 'boolean'], style: { color: '#7c6322' } },
    { types: ['operator'], style: { color: '#0d7377' } },
    { types: ['punctuation'], style: { color: '#4a5568' } },
    { types: ['property', 'constant'], style: { color: '#8b4e3a' } },
    { types: ['variable'], style: { color: '#2d3748' } },
  ],
};

function CodeBlock() {
  const [activeSdk, setActiveSdk] = useState('openai');
  const [langIndex, setLangIndex] = useState({openai: 0, anthropic: 0});
  const {colorMode} = useColorMode();
  const sdk = SDK_EXAMPLES[activeSdk];
  const activeIdx = langIndex[activeSdk];
  const prismLang = LANG_TO_PRISM[sdk.languages[activeIdx].lang] || 'bash';
  const theme = colorMode === 'dark' ? tidalDark : tidalLight;

  return (
    <div className={styles.codeBlock}>
      <div className={styles.codeHeader}>
        <div className={styles.sdkTabs}>
          {Object.entries(SDK_EXAMPLES).map(([key, val]) => (
            <button
              key={key}
              className={clsx(styles.sdkTab, activeSdk === key && styles.sdkTabActive)}
              onClick={() => setActiveSdk(key)}
            >
              {val.label}
            </button>
          ))}
        </div>
        <code className={styles.endpointLabel}>{sdk.endpoint}</code>
      </div>
      <div className={styles.langTabs}>
        {sdk.languages.map((ex, i) => (
          <button
            key={ex.lang}
            className={clsx(styles.langTab, i === activeIdx && styles.langTabActive)}
            onClick={() => setLangIndex(prev => ({...prev, [activeSdk]: i}))}
          >
            {ex.lang}
          </button>
        ))}
      </div>
      <Highlight theme={theme} code={sdk.languages[activeIdx].code} language={prismLang}>
        {({style, tokens, getLineProps, getTokenProps}) => (
          <pre className={styles.codeContent} style={style}>
            <code>
              {tokens.map((line, i) => (
                <div key={i} {...getLineProps({line})}>
                  {line.map((token, key) => (
                    <span key={key} {...getTokenProps({token})} />
                  ))}
                </div>
              ))}
            </code>
          </pre>
        )}
      </Highlight>
    </div>
  );
}

function Hero() {
  return (
    <section className={styles.hero}>
      <div className="container">
        <div className={styles.heroLayout}>
          <div className={styles.heroText}>
            <h1 className={styles.title}>
              Not a gateway.<br />
              The full stack.
            </h1>
            <p className={styles.subtitle}>
              Inference, vector stores, file storage, safety, tool calling,
              and agentic orchestration in a single OpenAI-compatible server.
              Pluggable providers, any language, deploy anywhere.
            </p>
            <InstallBlock />
            <div className={styles.actions}>
              <Link className={styles.primaryBtn} to="/docs/getting_started/quickstart">
                Get started
              </Link>
              <Link className={styles.secondaryBtn} to="/docs/api-openai">
                API docs
              </Link>
              <a className={styles.githubBtn} href="https://github.com/llamastack/llama-stack" target="_blank" rel="noopener noreferrer">
                GitHub
              </a>
            </div>
          </div>
          <div className={styles.heroCode}>
            <CodeBlock />
          </div>
        </div>
      </div>
    </section>
  );
}

function ApiSurface() {
  return (
    <section className={styles.apiSection}>
      <div className="container">
        <div className={styles.apiHeader}>
          <h2>Everything your AI app needs. One server.</h2>
          <p>
            More than inference routing. Llama Stack composes inference, storage,
            safety, and orchestration into a single process. Your agent can search
            a vector store, call a tool, check safety, and stream the response.
            No glue code. No sidecar services.
          </p>
        </div>
        <div className={styles.apiColumns}>
          {API_SURFACE.map(group => (
            <div key={group.category} className={styles.apiGroup}>
              <h3 className={styles.apiGroupTitle}>{group.category}</h3>
              <div className={styles.endpointList}>
                {group.endpoints.map(ep => (
                  <div key={ep.path} className={styles.endpointRow}>
                    <code className={styles.endpointPath}>{ep.path}</code>
                    <span className={styles.endpointName}>
                      {ep.label}
                      {ep.note && <span className={styles.endpointNote}>{ep.note}</span>}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
        <Link className={styles.textLink} to="/docs/api-openai">
          Full API reference
        </Link>
      </div>
    </section>
  );
}

function ServerNotLibrary() {
  return (
    <section className={styles.serverSection}>
      <div className="container">
        <div className={styles.serverLayout}>
          <div>
            <h2>A server, not a library</h2>
            <p>
              SDK abstractions couple your app to a specific language, release
              cycle, and import path. Llama Stack is an HTTP server. Your app
              talks to a standard API.
            </p>
            <p>
              Write in Python, Go, TypeScript, curl. Swap the server without
              touching application code. That's the difference between library
              abstraction and server abstraction.
            </p>
          </div>
          <div className={styles.serverComparison}>
            <div className={styles.comparisonRow}>
              <span className={styles.comparisonLabel}>SDK library</span>
              <code className={styles.comparisonCode}>from sdk import ...</code>
              <span className={styles.comparisonNote}>coupled</span>
            </div>
            <div className={styles.comparisonRow}>
              <span className={styles.comparisonLabel}>Llama Stack</span>
              <code className={styles.comparisonCode}>POST /v1/responses</code>
              <span className={styles.comparisonGood}>any language</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function Providers() {
  return (
    <section className={styles.providerSection}>
      <div className="container">
        <h2>23 inference providers. 13 vector stores. 7 safety backends.</h2>
        <p className={styles.providerDesc}>
          Develop locally with Ollama. Deploy to production with vLLM.
          Wrap Bedrock or Vertex without lock-in. Same API surface, different backend.
        </p>
        <div className={styles.providerGrid}>
          {PROVIDERS.map(p => (
            <Link key={p.name} className={styles.provider} to={p.href}>{p.name}</Link>
          ))}
        </div>
        <Link className={styles.textLink} to="/docs/providers">
          All providers
        </Link>
      </div>
    </section>
  );
}

function Architecture() {
  return (
    <section className={styles.archSection}>
      <div className="container">
        <h2>How it works</h2>
        <p className={styles.archDesc}>
          Your application talks to one server. That server routes
          to pluggable providers for inference, vector storage, files,
          safety, and tools. The composition happens at the server level,
          not in your application code.
        </p>
        <div className={styles.archImg}>
          <img src="/img/architecture-animated.svg" alt="Llama Stack Architecture" loading="lazy" />
        </div>
      </div>
    </section>
  );
}

function Bottom() {
  return (
    <section className={styles.bottomSection}>
      <div className="container">
        <div className={styles.bottomLayout}>
          <div>
            <h2>Open source</h2>
            <p>
              Apache 2.0 licensed. Contributions welcome.
            </p>
          </div>
          <div className={styles.bottomLinks}>
            <a href="https://github.com/llamastack/llama-stack" target="_blank" rel="noopener noreferrer">
              GitHub
            </a>
            <a href="https://discord.gg/llama-stack" target="_blank" rel="noopener noreferrer">
              Discord
            </a>
            <Link to="/docs/">
              Documentation
            </Link>
            <Link to="/blog">
              Blog
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  return (
    <Layout title="The Open-Source AI Application Server" description="Inference, vector stores, safety, tools, and agentic orchestration. One server, OpenAI-compatible, pluggable providers.">
      <main>
        <Hero />
        <ApiSurface />
        <ServerNotLibrary />
        <Providers />
        <Architecture />
        <Bottom />
      </main>
    </Layout>
  );
}

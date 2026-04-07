import React, {useState} from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import InstallBlock from '../components/InstallBlock';
import styles from './index.module.css';

const Icons = {
  github: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/>
    </svg>
  ),
  discord: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
      <path d="M20.317 4.37a19.791 19.791 0 00-4.885-1.515.074.074 0 00-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 00-5.487 0 12.64 12.64 0 00-.617-1.25.077.077 0 00-.079-.037A19.736 19.736 0 003.677 4.37a.07.07 0 00-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 00.031.057 19.9 19.9 0 005.993 3.03.078.078 0 00.084-.028c.462-.63.874-1.295 1.226-1.994a.076.076 0 00-.041-.106 13.107 13.107 0 01-1.872-.892.077.077 0 01-.008-.128 10.2 10.2 0 00.372-.292.074.074 0 01.077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 01.078.01c.12.098.246.198.373.292a.077.077 0 01-.006.127 12.299 12.299 0 01-1.873.892.077.077 0 00-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 00.084.028 19.839 19.839 0 006.002-3.03.077.077 0 00.032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 00-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.946 2.418-2.157 2.418z"/>
    </svg>
  ),
  docs: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 19.5A2.5 2.5 0 016.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z"/>
    </svg>
  ),
  blog: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 013 3L7 19l-4 1 1-4L16.5 3.5z"/>
    </svg>
  ),
  chat: (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
    </svg>
  ),
  zap: (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
    </svg>
  ),
  layers: (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>
    </svg>
  ),
  database: (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>
    </svg>
  ),
  file: (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/>
    </svg>
  ),
  cpu: (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>
    </svg>
  ),
  shield: (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
    </svg>
  ),
  message: (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/><polyline points="22,6 12,13 2,6"/>
    </svg>
  ),
  conversation: (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 11.5a8.38 8.38 0 01-.9 3.8 8.5 8.5 0 01-7.6 4.7 8.38 8.38 0 01-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 01-.9-3.8 8.5 8.5 0 014.7-7.6 8.38 8.38 0 013.8-.9h.5a8.48 8.48 0 018 8v.5z"/>
    </svg>
  ),
  plug: (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 22v-5"/><path d="M9 8V2"/><path d="M15 8V2"/><path d="M18 8v5a6 6 0 01-12 0V8z"/>
    </svg>
  ),
  stack: (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <rect x="4" y="2" width="16" height="6" rx="1"/><rect x="4" y="10" width="16" height="6" rx="1"/><rect x="4" y="18" width="16" height="4" rx="1"/>
    </svg>
  ),
  arrow: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>
    </svg>
  ),
};

const OPENAI_ENDPOINTS = [
  { icon: Icons.chat, label: 'Chat Completions', path: '/v1/chat/completions', desc: 'Chat and text completion endpoints', link: '/docs/api/inference' },
  { icon: Icons.zap, label: 'Responses', path: '/v1/responses', desc: 'Agentic orchestration with tool calling and MCP', link: '/docs/api/inference' },
  { icon: Icons.layers, label: 'Embeddings', path: '/v1/embeddings', desc: 'Text embeddings from any provider', link: '/docs/api/inference' },
  { icon: Icons.database, label: 'Vector Stores', path: '/v1/vector_stores', desc: 'Document storage and semantic search', link: '/docs/api/vector-io' },
  { icon: Icons.shield, label: 'Moderations', path: '/v1/moderations', desc: 'Content moderation and safety shields', link: '/docs/api/safety' },
  { icon: Icons.file, label: 'Files', path: '/v1/files', desc: 'File upload, processing, and extraction', link: '/docs/api/files' },
  { icon: Icons.stack, label: 'Batches', path: '/v1/batches', desc: 'Async batch processing at scale', link: '/docs/api/batches' },
  { icon: Icons.conversation, label: 'Conversations', path: '/v1/conversations', desc: 'Multi-turn conversation state and history', link: '/docs/api/conversations' },
  { icon: Icons.cpu, label: 'Models', path: '/v1/models', desc: 'Model discovery and management', link: '/docs/api/models' },
];

const ANTHROPIC_ENDPOINTS = [
  { icon: Icons.message, label: 'Messages API', path: '/v1/messages', desc: 'Chat completions with native Anthropic format', link: '/docs/api-openai/anthropic_messages' },
];

const NATIVE_ENDPOINTS = [
  { icon: Icons.plug, label: 'Connectors', path: '/v1/connectors', desc: 'External connectors like MCP servers', link: '/docs/api-experimental' },
  { icon: Icons.zap, label: 'Tools', path: '/v1/tools', desc: 'Tool discovery and runtime invocation', link: '/docs/api/list-tools-v-1-tools-get' },
];

const PROVIDERS = {
  inference: ['Ollama', 'vLLM', 'AWS Bedrock', 'Azure OpenAI', 'OpenAI', 'Anthropic', 'Gemini', '15+ more'],
  vector: ['PGVector', 'Qdrant', 'ChromaDB', 'Milvus', 'Weaviate', '4+ more'],
  tools: ['MCP Servers', 'Web Search', 'File Search (RAG)', 'PDF / Docling'],
};

const SDK_EXAMPLES = {
  openai: {
    label: 'OpenAI SDK',
    endpoint: '/v1/responses',
    languages: [
      {
        lang: 'Python',
        code: (s) => (
          <>
<span className={s.synKeyword}>from</span> <span className={s.synModule}>openai</span> <span className={s.synKeyword}>import</span> <span className={s.synClass}>OpenAI</span>{'\n'}
{'\n'}
<span className={s.synVar}>client</span> = <span className={s.synClass}>OpenAI</span>(<span className={s.synParam}>base_url</span>=<span className={s.synString}>"http://localhost:8321/v1"</span>, <span className={s.synParam}>api_key</span>=<span className={s.synString}>"fake"</span>){'\n'}
<span className={s.synVar}>response</span> = <span className={s.synVar}>client</span>.<span className={s.synMethod}>responses</span>.<span className={s.synMethod}>create</span>({'\n'}
{'    '}<span className={s.synParam}>model</span>=<span className={s.synString}>"llama-3.3-70b"</span>,{'\n'}
{'    '}<span className={s.synParam}>input</span>=<span className={s.synString}>"Summarize this repository"</span>,{'\n'}
{'    '}<span className={s.synParam}>tools</span>=[{'{'}<span className={s.synString}>"type"</span>: <span className={s.synString}>"web_search"</span>{'}'}],{'\n'}
)
          </>
        ),
      },
      {
        lang: 'curl',
        code: (s) => (
          <>
<span className={s.synVar}>curl</span> <span className={s.synString}>http://localhost:8321/v1/responses</span> \{'\n'}
{'  '}<span className={s.synParam}>-H</span> <span className={s.synString}>"Content-Type: application/json"</span> \{'\n'}
{'  '}<span className={s.synParam}>-d</span> <span className={s.synString}>'{'"'}{"{"}{'\n'}
{'    '}"model": "llama-3.3-70b",{'\n'}
{'    '}"input": "Summarize this repository",{'\n'}
{'    '}"tools": [{"{"}"type": "web_search"{"}"}]{'\n'}
{'  '}{"}"}'</span>
          </>
        ),
      },
      {
        lang: 'Node.js',
        code: (s) => (
          <>
<span className={s.synKeyword}>import</span> <span className={s.synClass}>OpenAI</span> <span className={s.synKeyword}>from</span> <span className={s.synString}>"openai"</span>;{'\n'}
{'\n'}
<span className={s.synKeyword}>const</span> <span className={s.synVar}>client</span> = <span className={s.synKeyword}>new</span> <span className={s.synClass}>OpenAI</span>({'{'}{'\n'}
{'  '}<span className={s.synParam}>baseURL</span>: <span className={s.synString}>"http://localhost:8321/v1"</span>,{'\n'}
{'  '}<span className={s.synParam}>apiKey</span>: <span className={s.synString}>"fake"</span>,{'\n'}
{'}'});{'\n'}
{'\n'}
<span className={s.synKeyword}>const</span> <span className={s.synVar}>response</span> = <span className={s.synKeyword}>await</span> <span className={s.synVar}>client</span>.<span className={s.synMethod}>responses</span>.<span className={s.synMethod}>create</span>({'{'}{'\n'}
{'  '}<span className={s.synParam}>model</span>: <span className={s.synString}>"llama-3.3-70b"</span>,{'\n'}
{'  '}<span className={s.synParam}>input</span>: <span className={s.synString}>"Summarize this repository"</span>,{'\n'}
{'  '}<span className={s.synParam}>tools</span>: [{'{'} <span className={s.synParam}>type</span>: <span className={s.synString}>"web_search"</span> {'}'}],{'\n'}
{'}'});
          </>
        ),
      },
      {
        lang: 'Go',
        code: (s) => (
          <>
<span className={s.synVar}>client</span> := <span className={s.synModule}>openai</span>.<span className={s.synClass}>NewClient</span>({'\n'}
{'  '}<span className={s.synModule}>option</span>.<span className={s.synMethod}>WithBaseURL</span>(<span className={s.synString}>"http://localhost:8321/v1"</span>),{'\n'}
{'  '}<span className={s.synModule}>option</span>.<span className={s.synMethod}>WithAPIKey</span>(<span className={s.synString}>"fake"</span>),{'\n'}
){'\n'}
{'\n'}
<span className={s.synVar}>response</span>, <span className={s.synVar}>err</span> := <span className={s.synVar}>client</span>.<span className={s.synMethod}>Responses</span>.<span className={s.synMethod}>New</span>({'\n'}
{'  '}<span className={s.synModule}>context</span>.<span className={s.synMethod}>TODO</span>(),{'\n'}
{'  '}<span className={s.synModule}>openai</span>.<span className={s.synClass}>ResponseNewParams</span>{'{'}{'\n'}
{'    '}<span className={s.synParam}>Model</span>: <span className={s.synString}>"llama-3.3-70b"</span>,{'\n'}
{'    '}<span className={s.synParam}>Input</span>: <span className={s.synString}>"Summarize this repository"</span>,{'\n'}
{'    '}<span className={s.synParam}>Tools</span>: []<span className={s.synClass}>ResponseToolUnionParam</span>{'{'}{'\n'}
{'      '}<span className={s.synModule}>openai</span>.<span className={s.synMethod}>WebSearchTool</span>(),{'\n'}
{'    '}{'}'},{'\n'}
{'  '}{'}'},{'\n'}
)
          </>
        ),
      },
    ],
  },
  anthropic: {
    label: 'Anthropic SDK',
    endpoint: '/v1/messages',
    languages: [
      {
        lang: 'Python',
        code: (s) => (
          <>
<span className={s.synKeyword}>from</span> <span className={s.synModule}>anthropic</span> <span className={s.synKeyword}>import</span> <span className={s.synClass}>Anthropic</span>{'\n'}
{'\n'}
<span className={s.synVar}>client</span> = <span className={s.synClass}>Anthropic</span>({'\n'}
{'    '}<span className={s.synParam}>base_url</span>=<span className={s.synString}>"http://localhost:8321/v1"</span>,{'\n'}
{'    '}<span className={s.synParam}>api_key</span>=<span className={s.synString}>"fake"</span>,{'\n'}
){'\n'}
<span className={s.synVar}>message</span> = <span className={s.synVar}>client</span>.<span className={s.synMethod}>messages</span>.<span className={s.synMethod}>create</span>({'\n'}
{'    '}<span className={s.synParam}>model</span>=<span className={s.synString}>"llama-3.3-70b"</span>,{'\n'}
{'    '}<span className={s.synParam}>max_tokens</span>=<span className={s.synVar}>1024</span>,{'\n'}
{'    '}<span className={s.synParam}>messages</span>=[{'\n'}
{'        '}{'{'}<span className={s.synString}>"role"</span>: <span className={s.synString}>"user"</span>,{'\n'}
{'         '}<span className={s.synString}>"content"</span>: <span className={s.synString}>"Summarize this repository"</span>{'}'}
{'\n'}{'    '}],{'\n'}
)
          </>
        ),
      },
      {
        lang: 'TypeScript',
        code: (s) => (
          <>
<span className={s.synKeyword}>import</span> <span className={s.synClass}>Anthropic</span> <span className={s.synKeyword}>from</span> <span className={s.synString}>"@anthropic-ai/sdk"</span>;{'\n'}
{'\n'}
<span className={s.synKeyword}>const</span> <span className={s.synVar}>client</span> = <span className={s.synKeyword}>new</span> <span className={s.synClass}>Anthropic</span>({'{'}{'\n'}
{'  '}<span className={s.synParam}>baseURL</span>: <span className={s.synString}>"http://localhost:8321/v1"</span>,{'\n'}
{'  '}<span className={s.synParam}>apiKey</span>: <span className={s.synString}>"fake"</span>,{'\n'}
{'}'});{'\n'}
{'\n'}
<span className={s.synKeyword}>const</span> <span className={s.synVar}>message</span> = <span className={s.synKeyword}>await</span> <span className={s.synVar}>client</span>.<span className={s.synMethod}>messages</span>.<span className={s.synMethod}>create</span>({'{'}{'\n'}
{'  '}<span className={s.synParam}>model</span>: <span className={s.synString}>"llama-3.3-70b"</span>,{'\n'}
{'  '}<span className={s.synParam}>max_tokens</span>: <span className={s.synVar}>1024</span>,{'\n'}
{'  '}<span className={s.synParam}>messages</span>: [{'{'} <span className={s.synParam}>role</span>: <span className={s.synString}>"user"</span>, <span className={s.synParam}>content</span>: <span className={s.synString}>"Summarize this repository"</span> {'}'}],{'\n'}
{'}'});
          </>
        ),
      },
    ],
  },
};

function CodeTabs() {
  const [activeSdk, setActiveSdk] = useState('openai');
  const [langIndex, setLangIndex] = useState({openai: 0, anthropic: 0});
  const sdk = SDK_EXAMPLES[activeSdk];
  const activeIdx = langIndex[activeSdk];

  const switchSdk = (key) => {
    setActiveSdk(key);
  };

  const switchLang = (idx) => {
    setLangIndex(prev => ({...prev, [activeSdk]: idx}));
  };

  return (
    <div className={styles.codeBlock}>
      <div className={styles.codeWindowDots}>
        <span /><span /><span />
      </div>
      <div className={styles.sdkToggle}>
        {Object.entries(SDK_EXAMPLES).map(([key, val]) => (
          <button
            key={key}
            className={clsx(styles.sdkBtn, activeSdk === key && styles.sdkBtnActive)}
            onClick={() => switchSdk(key)}
          >
            {val.label}
          </button>
        ))}
      </div>
      <div className={styles.codeSubHeader}>
        <div className={styles.codeTabs}>
          {sdk.languages.map((ex, i) => (
            <button
              key={ex.lang}
              className={clsx(styles.codeTab, i === activeIdx && styles.codeTabActive)}
              onClick={() => switchLang(i)}
            >
              {ex.lang}
            </button>
          ))}
        </div>
        <code className={styles.endpointBadge}>{sdk.endpoint}</code>
      </div>
      <pre><code>{sdk.languages[activeIdx].code(styles)}</code></pre>
    </div>
  );
}

function Hero() {
  return (
    <section className={styles.hero}>
      <div className={styles.heroMesh} />
      <div className={styles.heroGrid} />
      <div className="container">
        <div className={styles.heroInner}>
          <div className={styles.badge}>
            <span className={styles.badgeDot} />
            OpenAI + Anthropic API Server
          </div>
          <h1 className={styles.title}>
            Build AI apps with<br />
            <span className={styles.gradient}>any model, anywhere</span>
          </h1>
          <p className={styles.subtitle}>
            OpenAI and Anthropic compatible API server. Use any client, any framework,
            any model. Swap providers without changing code.
          </p>
          <InstallBlock />
          <div className={styles.actions}>
            <Link className={styles.primaryBtn} to="/docs/getting_started/quickstart">
              Get Started <span className={styles.btnArrow}>{Icons.arrow}</span>
            </Link>
            <Link className={styles.secondaryBtn} to="/docs/api-overview">API Reference</Link>
            <a className={styles.ghostBtn} href="https://github.com/llamastack/llama-stack" target="_blank" rel="noopener noreferrer">{Icons.github} GitHub</a>
          </div>
          <CodeTabs />
        </div>
      </div>
    </section>
  );
}

const PROVIDER_NAMES = [
  'Ollama', 'vLLM', 'OpenAI', 'Anthropic', 'AWS Bedrock',
  'Azure OpenAI', 'Gemini', 'Together AI', 'Fireworks',
  'PGVector', 'Qdrant', 'ChromaDB', 'Milvus', 'Weaviate',
];

function ProviderStrip() {
  const doubled = [...PROVIDER_NAMES, ...PROVIDER_NAMES];
  return (
    <section className={styles.providerStrip}>
      <div className="container">
        <p className={styles.stripLabel}>Works with</p>
      </div>
      <div className={styles.marqueeWrap}>
        <div className={styles.marqueeTrack}>
          {doubled.map((name, i) => (
            <span key={`${name}-${i}`} className={styles.stripItem}>{name}</span>
          ))}
        </div>
      </div>
      <div className={styles.stripMoreWrap}>
        <Link to="/docs/providers" className={styles.stripMore}>See all providers</Link>
      </div>
    </section>
  );
}

const STATS = [
  { value: '20+', label: 'Inference Providers' },
  { value: '11+', label: 'API Endpoints' },
  { value: '4', label: 'Client Languages' },
  { value: '100%', label: 'Open Source' },
];

function StatsRibbon() {
  return (
    <section className={styles.stats}>
      <div className="container">
        <div className={styles.statsGrid}>
          {STATS.map(s => (
            <div key={s.label} className={styles.statItem}>
              <span className={styles.statValue}>{s.value}</span>
              <span className={styles.statLabel}>{s.label}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function EndpointCard({endpoint}) {
  return (
    <div className={styles.card}>
      <Link to={endpoint.link} className={styles.cardLink}>
        <div className={styles.cardTop}>
          <div className={styles.cardIcon}>{endpoint.icon}</div>
          <code className={styles.path}>{endpoint.path}</code>
        </div>
        <h3>{endpoint.label}</h3>
        <p>{endpoint.desc}</p>
      </Link>
    </div>
  );
}

function Endpoints() {
  return (
    <section className={styles.endpoints}>
      <div className="container">
        <div className={styles.sectionHead}>
          <span className={styles.sectionTag}>OpenAI API</span>
          <h2>OpenAI-compatible endpoints</h2>
          <p>Use any OpenAI client library. Zero code changes.</p>
        </div>
        <div className={styles.grid}>
          {OPENAI_ENDPOINTS.map(f => <EndpointCard key={f.path} endpoint={f} />)}
        </div>
        <div className={styles.sectionHead}>
          <span className={styles.sectionTag}>Anthropic API</span>
          <h2>Anthropic-compatible endpoint</h2>
          <p>Use the Anthropic client library directly.</p>
        </div>
        <div className={styles.gridNative}>
          {ANTHROPIC_ENDPOINTS.map(f => <EndpointCard key={f.path} endpoint={f} />)}
        </div>
        <div className={styles.sectionHead}>
          <span className={styles.sectionTag}>Native APIs</span>
          <h2>Llama Stack native APIs</h2>
          <p>Additional endpoints beyond the OpenAI and Anthropic specs.</p>
        </div>
        <div className={styles.gridNative}>
          {NATIVE_ENDPOINTS.map(f => <EndpointCard key={f.path} endpoint={f} />)}
        </div>
      </div>
    </section>
  );
}

function Architecture() {
  return (
    <section className={styles.arch}>
      <div className="container">
        <div className={styles.sectionHead}>
          <span className={styles.sectionTag}>Architecture</span>
          <h2>How it works</h2>
          <p>OpenAI and Anthropic compatible, pluggable providers, deploy anywhere.</p>
        </div>
        <div className={styles.archImg}>
          <div className={styles.archGlow} />
          <img src="/img/architecture-animated.svg" alt="Llama Stack Architecture" loading="lazy" />
        </div>
      </div>
    </section>
  );
}

function ProviderSection() {
  return (
    <section className={styles.providers}>
      <div className="container">
        <div className={styles.sectionHead}>
          <span className={styles.sectionTag}>Providers</span>
          <h2>Plug in any provider</h2>
          <p>Develop locally with Ollama, deploy to production with vLLM or a managed service.</p>
        </div>
        <div className={styles.providerCols}>
          {Object.entries(PROVIDERS).map(([cat, items]) => (
            <div key={cat} className={styles.providerCol}>
              <h4>{cat === 'inference' ? 'Inference' : cat === 'vector' ? 'Vector Stores' : 'Tools'}</h4>
              <div className={styles.tags}>{items.map(n => <span key={n} className={styles.tag}>{n}</span>)}</div>
            </div>
          ))}
        </div>
        <div className={styles.providerLink}><Link to="/docs/providers">See all providers</Link></div>
      </div>
    </section>
  );
}

function Community() {
  return (
    <section className={styles.community}>
      <div className="container">
        <div className={styles.communityInner}>
          <span className={styles.sectionTag}>Community</span>
          <h2>Open source. Community driven.</h2>
          <p>Join thousands of developers building with Llama Stack</p>
          <div className={styles.links}>
            <a href="https://github.com/llamastack/llama-stack" className={styles.linkCard} target="_blank" rel="noopener noreferrer"><span className={styles.linkIcon}>{Icons.github}</span><strong>GitHub</strong><span>Star & contribute</span></a>
            <a href="https://discord.gg/llama-stack" className={styles.linkCard} target="_blank" rel="noopener noreferrer"><span className={styles.linkIcon}>{Icons.discord}</span><strong>Discord</strong><span>Chat with the community</span></a>
            <Link to="/docs/" className={styles.linkCard}><span className={styles.linkIcon}>{Icons.docs}</span><strong>Documentation</strong><span>Read the docs</span></Link>
            <Link to="/blog" className={styles.linkCard}><span className={styles.linkIcon}>{Icons.blog}</span><strong>Blog</strong><span>Latest updates</span></Link>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  return (
    <Layout title="OpenAI + Anthropic Compatible AI Server" description="OpenAI and Anthropic compatible API server. Any model, any infrastructure. Your data.">
      <main>
        <Hero />
        <ProviderStrip />
        <StatsRibbon />
        <Endpoints />
        <Architecture />
        <ProviderSection />
        <Community />
      </main>
    </Layout>
  );
}

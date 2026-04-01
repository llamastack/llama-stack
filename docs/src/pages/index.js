import React, {useState} from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
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
};

const FEATURES = [
  { label: 'Chat Completions', path: '/v1/chat/completions', desc: 'Standard OpenAI-compatible chat and completion endpoints' },
  { label: 'Responses API', path: '/v1/responses', desc: 'Server-side agentic orchestration with tool calling and MCP' },
  { label: 'Embeddings', path: '/v1/embeddings', desc: 'Text embeddings from any provider' },
  { label: 'Vector Stores', path: '/v1/vector_stores', desc: 'Managed document storage and semantic search' },
  { label: 'Files & Batches', path: '/v1/files', desc: 'File upload, processing, and batch operations' },
  { label: 'Models', path: '/v1/models', desc: 'Model discovery and management' },
];

const PROVIDERS = {
  inference: ['Ollama', 'vLLM', 'AWS Bedrock', 'Azure OpenAI', 'OpenAI', 'Anthropic', 'Gemini', '15+ more'],
  vector: ['PGVector', 'Qdrant', 'ChromaDB', 'Milvus', 'Weaviate', '4+ more'],
  tools: ['MCP Servers', 'Web Search', 'File Search (RAG)', 'PDF / Docling'],
};

const CODE_EXAMPLES = [
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
];

function CodeTabs() {
  const [active, setActive] = useState(0);
  return (
    <div className={styles.codeBlock}>
      <div className={styles.codeTabs}>
        {CODE_EXAMPLES.map((ex, i) => (
          <button
            key={ex.lang}
            className={clsx(styles.codeTab, i === active && styles.codeTabActive)}
            onClick={() => setActive(i)}
          >
            {ex.lang}
          </button>
        ))}
      </div>
      <pre><code>{CODE_EXAMPLES[active].code(styles)}</code></pre>
    </div>
  );
}

function Hero() {
  return (
    <section className={styles.hero}>
      <div className={styles.heroGlow} />
      <div className="container">
        <div className={styles.heroInner}>
          <div className={styles.badge}>OpenAI-Compatible API Server</div>
          <h1 className={styles.title}>
            Build AI apps with<br />
            <span className={styles.gradient}>any model, anywhere</span>
          </h1>
          <p className={styles.subtitle}>
            Drop-in replacement for the OpenAI API. Use any client, any framework,
            any model. Swap providers without changing code.
          </p>
          <div className={styles.actions}>
            <Link className={styles.primaryBtn} to="/docs/getting_started/quickstart">Get Started</Link>
            <Link className={styles.secondaryBtn} to="/docs/api-openai">API Reference</Link>
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
  return (
    <section className={styles.providerStrip}>
      <div className="container">
        <p className={styles.stripLabel}>Works with</p>
        <div className={styles.stripLogos}>
          {PROVIDER_NAMES.map(name => (
            <span key={name} className={styles.stripItem}>{name}</span>
          ))}
          <Link to="/docs/providers" className={styles.stripMore}>and more</Link>
        </div>
      </div>
    </section>
  );
}

function Endpoints() {
  return (
    <section className={styles.endpoints}>
      <div className="container">
        <div className={styles.sectionHead}><h2>OpenAI-compatible endpoints</h2><p>Use any OpenAI client library. Zero code changes.</p></div>
        <div className={styles.grid}>
          {FEATURES.map(f => (
            <div key={f.path} className={styles.card}>
              <code className={styles.path}>{f.path}</code>
              <h3>{f.label}</h3>
              <p>{f.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function Architecture() {
  return (
    <section className={styles.arch}>
      <div className="container">
        <div className={styles.sectionHead}><h2>How it works</h2><p>One API surface, pluggable providers, deploy anywhere</p></div>
        <div className={styles.archImg}><img src="/img/architecture-animated.svg" alt="Llama Stack Architecture" loading="lazy" /></div>
      </div>
    </section>
  );
}

function ProviderSection() {
  return (
    <section className={styles.providers}>
      <div className="container">
        <div className={styles.sectionHead}><h2>Plug in any provider</h2><p>Develop locally with Ollama, deploy to production with vLLM or a managed service</p></div>
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
    <Layout title="OpenAI-Compatible AI Server" description="Drop-in replacement for the OpenAI API. Any model, any infrastructure.">
      <main>
        <Hero />
        <ProviderStrip />
        <Endpoints />
        <Architecture />
        <ProviderSection />
        <Community />
      </main>
    </Layout>
  );
}

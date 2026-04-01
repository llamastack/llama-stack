import React from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import styles from './index.module.css';

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
            <a className={styles.ghostBtn} href="https://github.com/llamastack/llama-stack" target="_blank" rel="noopener noreferrer">GitHub</a>
          </div>
          <div className={styles.codeBlock}>
            <pre><code>{`from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")
response = client.chat.completions.create(
    model="llama-3.3-70b",
    messages=[{"role": "user", "content": "Hello"}],
)`}</code></pre>
          </div>
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
            <a href="https://github.com/llamastack/llama-stack" className={styles.linkCard} target="_blank" rel="noopener noreferrer"><strong>GitHub</strong><span>Star & contribute</span></a>
            <a href="https://discord.gg/llama-stack" className={styles.linkCard} target="_blank" rel="noopener noreferrer"><strong>Discord</strong><span>Chat with the community</span></a>
            <Link to="/docs/" className={styles.linkCard}><strong>Documentation</strong><span>Read the docs</span></Link>
            <Link to="/blog" className={styles.linkCard}><strong>Blog</strong><span>Latest updates</span></Link>
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
        <Endpoints />
        <Architecture />
        <ProviderSection />
        <Community />
      </main>
    </Layout>
  );
}

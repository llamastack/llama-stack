import React from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <div className={styles.heroContent}>
          <h1 className={styles.heroTitle}>Build AI Applications with Llama Stack</h1>
          <p className={styles.heroSubtitle}>
            Unified APIs for Inference, RAG, Agents, Tools, Safety, and Telemetry
          </p>
          <div className={styles.buttons}>
            <Link
              className={clsx('button button--primary button--lg', styles.getStartedButton)}
              to="/docs/getting_started/quickstart">
              🚀 Get Started
            </Link>
            <Link
              className={clsx('button button--primary button--lg', styles.apiButton)}
              to="/docs/api/llama-stack-specification">
              📚 API Reference
            </Link>
          </div>
        </div>
      </div>
    </header>
  );
}

function QuickStart() {
  return (
    <section className={styles.quickStart}>
      <div className="container">
        <div className="row">
          <div className="col col--6">
            <h2 className={styles.sectionTitle}>Quick Start</h2>
            <p className={styles.sectionDescription}>
              Get up and running with Llama Stack in just a few commands. Build your first RAG application locally.
            </p>
            <div className={styles.codeBlock}>
              <pre><code>{`# Install uv and start Ollama
ollama run llama3.2:3b --keepalive 60m

# Run Llama Stack server
OLLAMA_URL=http://localhost:11434 \\
  uv run --with llama-stack \\
  llama stack build --distro starter \\
  --image-type venv --run

# Try the Python SDK
from llama_stack_client import LlamaStackClient

client = LlamaStackClient(
  base_url="http://localhost:8321"
)

response = client.inference.chat_completion(
  model="Llama3.2-3B-Instruct",
  messages=[{
    "role": "user",
    "content": "What is machine learning?"
  }]
)`}</code></pre>
            </div>
          </div>
          <div className="col col--6">
            <h2 className={styles.sectionTitle}>Why Llama Stack?</h2>
            <div className={styles.features}>
              <div className={styles.feature}>
                <div className={styles.featureIcon}>🔗</div>
                <div>
                  <h4>Unified APIs</h4>
                  <p>One consistent interface for all your AI needs - inference, safety, agents, and more.</p>
                </div>
              </div>
              <div className={styles.feature}>
                <div className={styles.featureIcon}>🔄</div>
                <div>
                  <h4>Provider Flexibility</h4>
                  <p>Swap between providers without code changes. Start local, deploy anywhere.</p>
                </div>
              </div>
              <div className={styles.feature}>
                <div className={styles.featureIcon}>🛡️</div>
                <div>
                  <h4>Production Ready</h4>
                  <p>Built-in safety, monitoring, and evaluation tools for enterprise applications.</p>
                </div>
              </div>
              <div className={styles.feature}>
                <div className={styles.featureIcon}>📱</div>
                <div>
                  <h4>Multi-Platform</h4>
                  <p>SDKs for Python, Node.js, iOS, Android, and REST APIs for any language.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function CommunityLinks() {
  return (
    <section className={styles.community}>
      <div className="container">
        <div className={styles.communityContent}>
          <h2 className={styles.sectionTitle}>Join the Community</h2>
          <p className={styles.sectionDescription}>
            Connect with developers building the future of AI applications
          </p>
          <div className={styles.communityLinks}>
            <a
              href="https://github.com/llamastack/llama-stack"
              className={clsx('button button--outline button--lg', styles.communityButton)}
              target="_blank"
              rel="noopener noreferrer">
              <span className={styles.communityIcon}>⭐</span>
              Star on GitHub
            </a>
            <a
              href="https://discord.gg/llama-stack"
              className={clsx('button button--outline button--lg', styles.communityButton)}
              target="_blank"
              rel="noopener noreferrer">
              <span className={styles.communityIcon}>💬</span>
              Join Discord
            </a>
            <Link
              to="/docs/"
              className={clsx('button button--outline button--lg', styles.communityButton)}>
              <span className={styles.communityIcon}>📚</span>
              Read Docs
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title="Build AI Applications"
      description="The open-source framework for building generative AI applications with unified APIs for Inference, RAG, Agents, Tools, Safety, and Telemetry.">
      <HomepageHeader />
      <main>
        <QuickStart />
        <CommunityLinks />
      </main>
    </Layout>
  );
}

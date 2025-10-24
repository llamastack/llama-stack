import React, { useState, useEffect } from 'react';
import CodeBlock from '@theme/CodeBlock';

export default function CodeFromFile({
  src,
  language = 'python',
  title,
  startLine,
  endLine,
  highlightLines
}) {
  const [content, setContent] = useState('');
  const [error, setError] = useState(null);

  useEffect(() => {
    async function loadFile() {
      try {
        // File registration is now handled by the file-sync-plugin during build

        // Load file from static/imported-files directory
        const response = await fetch(`/imported-files/${src}`);
        if (!response.ok) {
          throw new Error(`Failed to fetch: ${response.status}`);
        }
        let text = await response.text();

        // Handle line range if specified (filtering is done at build time)
        if (startLine || endLine) {
          const lines = text.split('\n');
          const start = startLine ? Math.max(0, startLine - 1) : 0;
          const end = endLine ? Math.min(lines.length, endLine) : lines.length;
          text = lines.slice(start, end).join('\n');
        }

        setContent(text);
      } catch (err) {
        console.error('Failed to load file:', err);
        setError(`Failed to load ${src}: ${err.message}`);
      }
    }

    loadFile();
  }, [src, startLine, endLine]);

  if (error) {
    return <div style={{ color: 'red', padding: '1rem', border: '1px solid red', borderRadius: '4px' }}>
      Error: {error}
    </div>;
  }

  if (!content) {
    return <div>Loading {src}...</div>;
  }

  // Auto-detect language from file extension if not provided
  const detectedLanguage = language || getLanguageFromExtension(src);

  return (
    <CodeBlock
      language={detectedLanguage}
      title={title || src}
      metastring={highlightLines ? `{${highlightLines}}` : undefined}
    >
      {content}
    </CodeBlock>
  );
}

function getLanguageFromExtension(filename) {
  const ext = filename.split('.').pop();
  const languageMap = {
    'py': 'python',
    'js': 'javascript',
    'jsx': 'jsx',
    'ts': 'typescript',
    'tsx': 'tsx',
    'md': 'markdown',
    'sh': 'bash',
    'yaml': 'yaml',
    'yml': 'yaml',
    'json': 'json',
    'css': 'css',
    'html': 'html',
    'cpp': 'cpp',
    'c': 'c',
    'java': 'java',
    'go': 'go',
    'rs': 'rust',
    'php': 'php',
    'rb': 'ruby',
  };
  return languageMap[ext] || 'text';
}

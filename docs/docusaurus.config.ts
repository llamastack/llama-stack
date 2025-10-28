// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

import type * as Preset from "@docusaurus/preset-classic";
import type { Config } from "@docusaurus/types";
import type * as Plugin from "@docusaurus/types/src/plugin";
import type * as OpenApiPlugin from "docusaurus-plugin-openapi-docs";

const config: Config = {
  title: 'Llama Stack',
  tagline: 'The open-source framework for building generative AI applications',
  url: 'https://llamastack.github.io',
  baseUrl: '/',
  onBrokenLinks: "warn",
  onBrokenMarkdownLinks: "warn",
  favicon: "img/favicon.ico",

  // Enhanced favicon and meta configuration
  headTags: [
    {
      tagName: 'link',
      attributes: {
        rel: 'icon',
        type: 'image/png',
        sizes: '32x32',
        href: '/img/favicon-32x32.png',
      },
    },
    {
      tagName: 'link',
      attributes: {
        rel: 'icon',
        type: 'image/png',
        sizes: '16x16',
        href: '/img/favicon-16x16.png',
      },
    },
    {
      tagName: 'link',
      attributes: {
        rel: 'apple-touch-icon',
        sizes: '180x180',
        href: '/img/llama-stack-logo.png',
      },
    },
    {
      tagName: 'meta',
      attributes: {
        name: 'theme-color',
        content: '#7C3AED', // Purple color from your logo
      },
    },
    {
      tagName: 'link',
      attributes: {
        rel: 'manifest',
        href: '/site.webmanifest',
      },
    },
  ],

  // GitHub pages deployment config.
  organizationName: 'reluctantfuturist',
  projectName: 'llama-stack',
  trailingSlash: false,

  presets: [
    [
      "classic",
      {
        docs: {
          sidebarPath: require.resolve("./sidebars.ts"),
          docItemComponent: "@theme/ApiItem", // Derived from docusaurus-theme-openapi
          remarkPlugins: [
            [require('remark-code-import'), {
              rootDir: require('path').join(__dirname, '..') // Repository root
            }]
          ],
        },
        blog: false,
        theme: {
          customCss: require.resolve("./src/css/custom.css"),
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/llama-stack.png',
    navbar: {
      title: 'Llama Stack',
      logo: {
        alt: 'Llama Stack Logo',
        src: 'img/llama-stack-logo.png',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          type: 'dropdown',
          label: 'API Reference',
          position: 'left',
          to: '/docs/api-overview',
          items: [
            {
              type: 'docSidebar',
              sidebarId: 'stableApiSidebar',
              label: '🟢 Stable APIs',
            },
            {
              type: 'docSidebar',
              sidebarId: 'experimentalApiSidebar',
              label: '🟡 Experimental APIs',
            },
            {
              type: 'docSidebar',
              sidebarId: 'deprecatedApiSidebar',
              label: '🔴 Deprecated APIs',
            },
          ],
        },
        {
          href: 'https://github.com/llamastack/llama-stack',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'Getting Started',
              to: '/docs/getting_started/quickstart',
            },
            {
              label: 'Concepts',
              to: '/docs/concepts',
            },
            {
              label: 'API Reference',
              to: '/docs/api-overview',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'Discord',
              href: 'https://discord.gg/llama-stack',
            },
            {
              label: 'GitHub Discussions',
              href: 'https://github.com/llamastack/llama-stack/discussions',
            },
            {
              label: 'Issues',
              href: 'https://github.com/llamastack/llama-stack/issues',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/llamastack/llama-stack',
            },
            {
              label: 'PyPI',
              href: 'https://pypi.org/project/llama-stack/',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Meta Platforms, Inc. Built with Docusaurus.`,
    },
    prism: {
      additionalLanguages: [
        'ruby',
        'csharp',
        'php',
        'java',
        'powershell',
        'json',
        'bash',
        'python',
        'yaml',
      ],
    },
    docs: {
      sidebar: {
        hideable: true,
      },
    },
    // Language tabs for API documentation
    languageTabs: [
      {
        highlight: "python",
        language: "python",
        logoClass: "python",
      },
      {
        highlight: "bash",
        language: "curl",
        logoClass: "curl",
      },
      {
        highlight: "javascript",
        language: "nodejs",
        logoClass: "nodejs",
      },
      {
        highlight: "java",
        language: "java",
        logoClass: "java",
      },
    ],
  } satisfies Preset.ThemeConfig,

  plugins: [
    [
      "docusaurus-plugin-openapi-docs",
      {
        id: "openapi",
        docsPluginId: "classic",
        config: {
          stable: {
            specPath: "static/llama-stack-spec.yaml",
            outputDir: "docs/api",
            downloadUrl: "https://raw.githubusercontent.com/meta-llama/llama-stack/main/docs/static/llama-stack-spec.yaml",
            sidebarOptions: {
              groupPathsBy: "tag",
              categoryLinkSource: "tag",
            },
          } satisfies OpenApiPlugin.Options,
          experimental: {
            specPath: "static/experimental-llama-stack-spec.yaml",
            outputDir: "docs/api-experimental",
            downloadUrl: "https://raw.githubusercontent.com/meta-llama/llama-stack/main/docs/static/experimental-llama-stack-spec.yaml",
            sidebarOptions: {
              groupPathsBy: "tag",
              categoryLinkSource: "tag",
            },
          } satisfies OpenApiPlugin.Options,
          deprecated: {
            specPath: "static/deprecated-llama-stack-spec.yaml",
            outputDir: "docs/api-deprecated",
            downloadUrl: "https://raw.githubusercontent.com/meta-llama/llama-stack/main/docs/static/deprecated-llama-stack-spec.yaml",
            sidebarOptions: {
              groupPathsBy: "tag",
              categoryLinkSource: "tag",
            },
          } satisfies OpenApiPlugin.Options,
        } satisfies Plugin.PluginOptions,
      },
    ],
  ],

  themes: [
    "docusaurus-theme-openapi-docs",
    [
      require.resolve("@easyops-cn/docusaurus-search-local"),
      {
        // Optimization for production
        hashed: true,

        // Language settings
        language: ["en"],

        // Content indexing settings
        indexDocs: true,
        indexBlog: false, // No blog in Llama Stack
        indexPages: true,

        // Route configuration
        docsRouteBasePath: '/docs',

        // Search behavior optimization for technical docs
        searchResultLimits: 8,
        searchResultContextMaxLength: 50,
        explicitSearchResultPath: true,

        // User experience enhancements
        searchBarShortcut: true,
        searchBarShortcutHint: true,
        searchBarPosition: "right",

        // Performance optimizations
        ignoreFiles: [
          "node_modules/**/*",
        ],
      },
    ],
  ],
};

export default config;

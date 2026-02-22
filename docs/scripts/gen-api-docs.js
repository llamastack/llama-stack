#!/usr/bin/env node

/**
 * API Documentation Generation Script with Hash-Based Caching
 *
 * This script wraps the docusaurus gen-api-docs command with caching logic
 * to skip regeneration when OpenAPI specs haven't changed.
 *
 * Usage:
 *   node scripts/gen-api-docs.js [all|stable|experimental|deprecated] [--force]
 *
 * Options:
 *   --force    Force regeneration even if specs haven't changed
 *   --parallel Run generation for multiple specs in parallel (when using 'all')
 */

const { execSync, spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const DOCS_DIR = path.join(__dirname, '..');
const CACHE_FILE = path.join(DOCS_DIR, '.api-docs-cache.json');

// OpenAPI spec configurations matching docusaurus.config.ts
const SPECS = {
  stable: {
    specPath: path.join(DOCS_DIR, 'static', 'llama-stack-spec.yaml'),
    outputDir: path.join(DOCS_DIR, 'docs', 'api'),
  },
  experimental: {
    specPath: path.join(DOCS_DIR, 'static', 'experimental-llama-stack-spec.yaml'),
    outputDir: path.join(DOCS_DIR, 'docs', 'api-experimental'),
  },
  deprecated: {
    specPath: path.join(DOCS_DIR, 'static', 'deprecated-llama-stack-spec.yaml'),
    outputDir: path.join(DOCS_DIR, 'docs', 'api-deprecated'),
  },
};

/**
 * Compute SHA-256 hash of a file
 */
function hashFile(filePath) {
  if (!fs.existsSync(filePath)) {
    return null;
  }
  const content = fs.readFileSync(filePath);
  return crypto.createHash('sha256').update(content).digest('hex');
}

/**
 * Load the cache file
 */
function loadCache() {
  if (!fs.existsSync(CACHE_FILE)) {
    return {};
  }
  try {
    return JSON.parse(fs.readFileSync(CACHE_FILE, 'utf8'));
  } catch (error) {
    console.warn('Warning: Could not read cache file, starting fresh');
    return {};
  }
}

/**
 * Save the cache file
 */
function saveCache(cache) {
  fs.writeFileSync(CACHE_FILE, JSON.stringify(cache, null, 2));
}

/**
 * Check if a spec needs regeneration
 */
function needsRegeneration(specName, cache, force) {
  if (force) {
    return true;
  }

  const spec = SPECS[specName];
  if (!spec) {
    console.error(`Unknown spec: ${specName}`);
    return true;
  }

  const currentHash = hashFile(spec.specPath);
  if (!currentHash) {
    console.warn(`Warning: Spec file not found: ${spec.specPath}`);
    return true;
  }

  const cachedHash = cache[specName]?.hash;
  const outputExists = fs.existsSync(spec.outputDir) &&
    fs.readdirSync(spec.outputDir).length > 0;

  if (!outputExists) {
    console.log(`  ${specName}: Output directory empty or missing, will regenerate`);
    return true;
  }

  if (currentHash !== cachedHash) {
    console.log(`  ${specName}: Spec file changed, will regenerate`);
    return true;
  }

  console.log(`  ${specName}: No changes detected, skipping`);
  return false;
}

/**
 * Generate docs for a single spec
 */
function generateDocs(specName) {
  return new Promise((resolve, reject) => {
    console.log(`Generating API docs for: ${specName}`);
    const startTime = Date.now();

    const child = spawn('npm', ['run', 'docusaurus', 'gen-api-docs', specName], {
      cwd: DOCS_DIR,
      stdio: 'inherit',
      shell: true,
    });

    child.on('close', (code) => {
      const duration = ((Date.now() - startTime) / 1000).toFixed(1);
      if (code === 0) {
        console.log(`Completed ${specName} in ${duration}s`);
        resolve({ specName, success: true, duration });
      } else {
        console.error(`Failed to generate ${specName} (exit code: ${code})`);
        reject(new Error(`Generation failed for ${specName}`));
      }
    });

    child.on('error', (error) => {
      reject(error);
    });
  });
}

/**
 * Generate docs for a single spec (synchronous version)
 */
function generateDocsSync(specName) {
  console.log(`Generating API docs for: ${specName}`);
  const startTime = Date.now();

  try {
    execSync(`npm run docusaurus gen-api-docs ${specName}`, {
      cwd: DOCS_DIR,
      stdio: 'inherit',
    });
    const duration = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(`Completed ${specName} in ${duration}s`);
    return true;
  } catch (error) {
    console.error(`Failed to generate ${specName}`);
    return false;
  }
}

/**
 * Main function
 */
async function main() {
  const args = process.argv.slice(2);
  const force = args.includes('--force');
  const parallel = args.includes('--parallel');
  const target = args.find(arg => !arg.startsWith('--')) || 'all';

  console.log('API Documentation Generator with Caching');
  console.log('========================================');
  console.log(`Target: ${target}, Force: ${force}, Parallel: ${parallel}`);
  console.log('');

  const cache = loadCache();
  const specsToGenerate = [];

  if (target === 'all') {
    // Check all specs
    console.log('Checking which specs need regeneration...');
    for (const specName of Object.keys(SPECS)) {
      if (needsRegeneration(specName, cache, force)) {
        specsToGenerate.push(specName);
      }
    }
  } else if (SPECS[target]) {
    if (needsRegeneration(target, cache, force)) {
      specsToGenerate.push(target);
    }
  } else {
    console.error(`Unknown target: ${target}`);
    console.error('Valid targets: all, stable, experimental, deprecated');
    process.exit(1);
  }

  console.log('');

  if (specsToGenerate.length === 0) {
    console.log('All API docs are up to date. Nothing to generate.');
    console.log('Use --force to regenerate anyway.');
    return;
  }

  console.log(`Generating ${specsToGenerate.length} spec(s): ${specsToGenerate.join(', ')}`);
  console.log('');

  const startTime = Date.now();
  let success = true;

  if (parallel && specsToGenerate.length > 1) {
    // Parallel generation
    console.log('Running in parallel mode...');
    try {
      const results = await Promise.all(specsToGenerate.map(generateDocs));
      for (const result of results) {
        if (result.success) {
          cache[result.specName] = {
            hash: hashFile(SPECS[result.specName].specPath),
            generatedAt: new Date().toISOString(),
          };
        }
      }
    } catch (error) {
      console.error('Parallel generation failed:', error.message);
      success = false;
    }
  } else {
    // Sequential generation
    for (const specName of specsToGenerate) {
      if (generateDocsSync(specName)) {
        cache[specName] = {
          hash: hashFile(SPECS[specName].specPath),
          generatedAt: new Date().toISOString(),
        };
      } else {
        success = false;
      }
    }
  }

  saveCache(cache);

  const totalDuration = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log('');
  console.log(`Total time: ${totalDuration}s`);

  if (!success) {
    process.exit(1);
  }
}

main().catch((error) => {
  console.error('Error:', error.message);
  process.exit(1);
});

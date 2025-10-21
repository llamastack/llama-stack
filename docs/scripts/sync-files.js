#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

// Repository root is always one level up from docs
const repoRoot = path.join(__dirname, '..', '..');

// Get all requested files from the usage tracking file
function getRequestedFiles() {
  const usageFile = path.join(__dirname, '..', 'static', 'imported-files', 'usage.json');
  if (!fs.existsSync(usageFile)) {
    return [];
  }

  try {
    const usage = JSON.parse(fs.readFileSync(usageFile, 'utf8'));
    return usage.files || [];
  } catch (error) {
    console.warn('Could not read usage file:', error.message);
    return [];
  }
}

// Track file usage
function trackFileUsage(filePath) {
  const usageFile = path.join(__dirname, '..', 'static', 'imported-files', 'usage.json');
  const usageDir = path.dirname(usageFile);

  // Ensure directory exists
  if (!fs.existsSync(usageDir)) {
    fs.mkdirSync(usageDir, { recursive: true });
  }

  let usage = { files: [] };
  if (fs.existsSync(usageFile)) {
    try {
      usage = JSON.parse(fs.readFileSync(usageFile, 'utf8'));
    } catch (error) {
      console.warn('Could not read existing usage file, creating new one');
    }
  }

  if (!usage.files.includes(filePath)) {
    usage.files.push(filePath);
    fs.writeFileSync(usageFile, JSON.stringify(usage, null, 2));
  }
}

// Filter content based on file type and options
function filterContent(content, filePath) {
  let lines = content.split('\n');

  // Skip copyright header for Python files
  if (filePath.endsWith('.py')) {
    // Read the license header file
    const licenseHeaderPath = path.join(repoRoot, 'docs', 'license_header.txt');
    if (fs.existsSync(licenseHeaderPath)) {
      try {
        const licenseText = fs.readFileSync(licenseHeaderPath, 'utf8');
        const licenseLines = licenseText.trim().split('\n');

        // Check if file starts with the license header (accounting for # comments)
        if (lines.length >= licenseLines.length) {
          let matches = true;
          for (let i = 0; i < licenseLines.length; i++) {
            const codeLine = lines[i]?.replace(/^#\s*/, '').trim();
            const licenseLine = licenseLines[i]?.trim();
            if (codeLine !== licenseLine) {
              matches = false;
              break;
            }
          }

          if (matches) {
            // Skip the license header and any trailing empty lines
            let skipTo = licenseLines.length;
            while (skipTo < lines.length && lines[skipTo].trim() === '') {
              skipTo++;
            }
            lines = lines.slice(skipTo);
          }
        }
      } catch (error) {
        console.warn(`Could not read license header, skipping filtering for ${filePath}`);
      }
    }
  }

  // Trim empty lines from start and end
  while (lines.length > 0 && lines[0].trim() === '') {
    lines.shift();
  }
  while (lines.length > 0 && lines[lines.length - 1].trim() === '') {
    lines.pop();
  }

  return lines.join('\n');
}

// Sync a file from repo root to static directory
function syncFile(filePath) {
  const sourcePath = path.join(repoRoot, filePath);
  const destPath = path.join(__dirname, '..', 'static', 'imported-files', filePath);
  const destDir = path.dirname(destPath);

  // Ensure destination directory exists
  if (!fs.existsSync(destDir)) {
    fs.mkdirSync(destDir, { recursive: true });
  }

  try {
    if (fs.existsSync(sourcePath)) {
      const content = fs.readFileSync(sourcePath, 'utf8');
      const filteredContent = filterContent(content, filePath);
      fs.writeFileSync(destPath, filteredContent);
      console.log(`✅ Synced ${filePath}`);
      trackFileUsage(filePath);
      return true;
    } else {
      console.warn(`⚠️  Source file not found: ${sourcePath}`);
      return false;
    }
  } catch (error) {
    console.error(`❌ Error syncing ${filePath}:`, error.message);
    return false;
  }
}

// Main execution
console.log(`📁 Repository root: ${path.resolve(repoRoot)}`);

// Get files that are being requested by the documentation
const requestedFiles = getRequestedFiles();
console.log(`📄 Syncing ${requestedFiles.length} requested files...`);

if (requestedFiles.length === 0) {
  console.log('ℹ️  No files requested yet. Files will be synced when first referenced in documentation.');
} else {
  requestedFiles.forEach(filePath => {
    syncFile(filePath);
  });
}

console.log('✅ File sync complete!');

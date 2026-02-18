# Llama Stack Release Process

This document outlines the release process for Llama Stack, providing predictability for the community on feature delivery timelines and release expectations.

## Release Schedule

Llama Stack follows [Semantic Versioning](https://semver.org/) with three release streams:

| Release Type | Cadence | Description |
|-------------|---------|-------------|
| **Major (X.0.0)** | Every 6-8 months | Breaking changes, major new features, architectural changes |
| **Minor (0.Y.0)** | Monthly | New features, non-breaking API additions, significant improvements |
| **Patch (0.0.Z)** | Weekly | Bug fixes, security patches, documentation updates |

## Version Numbering

Releases follow the `X.Y.Z` pattern:
- **X (Major)**: Incremented for breaking changes or significant architectural updates
- **Y (Minor)**: Incremented for new features and non-breaking enhancements
- **Z (Patch)**: Incremented for bug fixes and minor improvements

### Release Candidates

For minor and major releases, release candidates (RC) are published before the final release:
- Format: `vX.Y.ZrcN` (e.g., `v0.4.0rc1`, `v0.4.0rc2`)
- Python RC packages are published to test.pypi for community testing
- Multiple RCs may be issued until the release is stable

## Branching Strategy

- **`main`**: Active development branch, always contains the latest code
- **`release-X.Y.x`**: Release branches for each minor version (e.g., `release-0.3.x`, `release-0.4.x`)
- Patch releases are made from release branches
- Critical fixes are backported from `main` to active release branches using Mergify

## Milestone Management

### Tracking Work

- **Issues only**: Add only issues to milestones, not PRs (avoids duplicate tracking)
- **Milestone creation**: Create milestones for each planned minor and major release
- **Small fixes**: Quick-landing PRs for small fixes don't require milestone tracking

### Release Criteria

A version is released when:
1. All issues in the corresponding milestone are completed, **OR**
2. Remaining issues are moved to a future milestone with documented rationale

### Triaging

- Triagers manage milestones and prioritize issues
- Discussions happen in the `#triage` Discord channel
- Priority decisions are reviewed in community calls

## Release Process

### Release Owner

Each release has a designated **Release Owner** from the [CODEOWNERS](./CODEOWNERS) group who is responsible for:

1. Creating a dedicated Discord thread in `#release` channel
2. Coordinating testing activities
3. Managing the release timeline
4. Publishing release artifacts
5. Announcing the release

### Testing Requirements

Testing requirements scale with release type:

#### Patch Releases (Z-stream)
- Rely primarily on automated CI tests
- Quick turnaround for critical fixes
- Manual verification only for specific fix validation

#### Minor Releases (Y-stream)
- Automated CI tests must pass
- Manual feature testing for new functionality
- Documentation verification
- **Community testing window: 1 week**
- Release candidates published for community validation

#### Major Releases (X-stream)
- Comprehensive automated test suite
- Scheduled testing period with predefined test plans
- Cross-provider compatibility testing
- Performance benchmarking
- **Community testing window: 2-3 weeks**
- Multiple release candidates as needed

### Release Checklist

For each release, the Release Owner should complete:

- [ ] Create release-specific thread in `#releases` Discord channel
- [ ] Complete the technical release steps below
- [ ] Generate release notes
- [ ] Announce in `#announcements` Discord channel

### Technical Release Steps

#### Patch release (e.g., 0.4.5 on existing `release-0.4.x`)

**Pre-release on `release-0.4.x`:**

- [ ] Cherry-pick any needed fixes to the release branch
- [ ] Update `fallback_version` to `"0.4.5"` in both:
  - `pyproject.toml`
  - `src/llama_stack_api/pyproject.toml`
- [ ] Update `llama-stack-client==0.4.4` → `llama-stack-client==0.4.5` in `pyproject.toml` (2 places: `[project.optional-dependencies]` and `[dependency-groups]`)
- [ ] PR and merge to `release-0.4.x` (CI will fail on unresolvable client pin — that's expected)

**Release:**

- [ ] Create GitHub release: tag `v0.4.5`, target `release-0.4.x`
- [ ] Verify all 4 packages published (llama-stack, llama-stack-api, llama-stack-client python+typescript)

**Post-release on `release-0.4.x`:**

- [ ] `cd src/llama_stack_ui && npm install llama-stack-client@^0.4.5`
- [ ] Commit updated `package.json` + `package-lock.json`

**Post-release on `main`:**

- [ ] Update `fallback_version` to `"0.4.6.dev0"` in both `pyproject.toml` files

#### Minor release (e.g., 0.5.0 — new release branch)

**All of the above, plus:**

- [ ] Create `release-0.5.x` branch off `main`
- [ ] Ensure the release branch has the setuptools-scm config in both `pyproject.toml` files (`dynamic = ["version"]`, `[tool.setuptools_scm]`, etc.)
- [ ] Post-release `fallback_version` on `main` becomes `"0.5.1.dev0"`

## Release Artifacts

Each release includes:

- **PyPI package**: `llama-stack` and `llama-stack-client`
- **npm package**: `llama-stack-client`
- **Docker images**: Distribution images on Docker Hub
- **GitHub Release**: Tagged release with release notes
- **Documentation**: Updated docs at https://llamastack.github.io

See [CONTRIBUTING.md](./CONTRIBUTING.md) for general contribution guidelines.

## Maintenance Policy

Llama Stack actively maintains the **last 2 stable minor releases**.

### What "Maintained" Means

- **Bug fixes**: Critical bugs are backported to maintained release branches
- **Security patches**: Security vulnerabilities are patched in maintained releases
- **Patch releases (Z-stream)**: Maintained releases receive regular patch releases

### Maintenance Timeline

| Release | Status | Notes |
|---------|--------|-------|
| Current minor (0.Y.0) | ✅ Actively maintained | Bug fixes and security patches |
| Previous minor (0.Y-1.0) | ✅ Maintained | Bug fixes and security patches |
| Older releases | ❌ Unmaintained | No backports; upgrade recommended |

### Example

If the current release is `v0.4.x`:
- `v0.4.x` — Actively maintained (current)
- `v0.3.x` — Maintained (bug fixes only)
- `v0.2.x` and earlier — Unmaintained

Users on unmaintained versions are encouraged to upgrade to continue receiving fixes.

## How the Release Workflow Works

The unified workflow (`.github/workflows/pypi.yml`) builds and publishes all packages:

- **Local packages** (llama-stack, llama-stack-api): version comes from the git tag via `SETUPTOOLS_SCM_PRETEND_VERSION`
- **External packages** (llama-stack-client python/typescript): the workflow patches `pyproject.toml`/`_version.py`/`package.json` at build time using the tag version via `sed`/`npm version`
- `fallback_version` is only used for nightly/dev builds and Docker — not for releases
- The workflow always runs from `main` but checks out the tag's commit for local packages

### Workflow Modes

| Trigger | Version | Target |
|---|---|---|
| `release: published` | From tag (`v0.4.5` → `0.4.5`) | pypi.org + npm |
| `schedule` (nightly) | `{fallback_base}.dev{YYYYMMDD}` | test.pypi.org |
| `workflow_dispatch` dry_run=test-pypi | `{fallback_base}.dev{YYYYMMDD}` or manual `version` input | test.pypi.org |
| `workflow_dispatch` dry_run=off | Manual `version` input | pypi.org + npm |
| `workflow_dispatch` dry_run=build-only | N/A | No publish |

## Future Improvements

### 1. Automate the version bumps

Create a `prepare-release.sh` script or GitHub Action that:
- Takes a version as input (e.g., `0.4.5`)
- Updates `fallback_version` in both `pyproject.toml` files
- Updates `llama-stack-client` pins in `pyproject.toml`
- Opens a PR to the release branch

This eliminates the manual file edits and the "CI will fail" problem since it can be sequenced.

### 2. Remove the client pin problem

The `llama-stack-client==X.Y.Z` pin in `pyproject.toml` can't be satisfied until the client is published, but the client is published in the same workflow run. Options:
- Change the pin to `>=X.Y.Z` or `~=X.Y` so it doesn't require an exact match that doesn't exist yet
- Remove the pin from the release branch entirely and let the workflow handle compatibility
- Publish client packages first in a separate step, then update pins, then publish llama-stack

### 3. Post-release lockfile update should be automated

After a release, the npm lockfile update (`npm install` in `llama_stack_ui`) is a manual step that's easy to forget. A post-release GitHub Action triggered by `release: published` could do this automatically and open a PR.

### 4. Post-release fallback bump on `main` should be automated

Same idea — a post-release action bumps `fallback_version` to the next `.dev0` on `main` and opens a PR.

### 5. Let setuptools-scm infer version from tags directly

Right now the workflow computes the version separately and passes it via `SETUPTOOLS_SCM_PRETEND_VERSION`. This exists because version tags live on release branches, and `main` has no tags for setuptools-scm to find.

The ideal setup: tag `main` (or ensure tags are reachable from the build checkout) so setuptools-scm just works natively. This would:
- Eliminate the `compute-version` step entirely
- Eliminate `fallback_version` management (no more bumping it post-release)
- Make `uv build` work correctly locally without any env vars
- Let setuptools-scm generate dev versions automatically (e.g., `0.4.4.dev3+gabcdef` based on commits since last tag)

**This is the highest-impact improvement because it eliminates the need for items 1 and 4 above.** No more `fallback_version` means no post-release bump on `main` (#4), and no version fields to update means no prepare-release script (#1). The release checklist shrinks to: cherry-pick fixes, update client pins, tag, done.

The blocker is that tags currently only exist on release branches. If we tagged `main` at release time too (or merged release branches back), setuptools-scm would have the ancestry it needs. Alternatively, the CI checkout could fetch tags from all branches (`git fetch --tags`) so setuptools-scm can find the nearest tag.

### 6. Client repos should use dynamic versioning

The `llama-stack-client-python` and `llama-stack-client-typescript` repos use static versions. The workflow patches them with `sed` at build time, which is fragile. If those repos adopted setuptools-scm (Python) or a similar scheme, the workflow could just set an env var instead of rewriting files.

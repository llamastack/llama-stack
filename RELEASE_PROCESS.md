# Llama Stack Release Process

This document outlines the release process for Llama Stack, providing predictability for the community on feature delivery timelines and release expectations.

## Release Schedule

Llama Stack follows [Semantic Versioning](https://semver.org/) with three release streams:

| Release Type | Cadence | Description |
|-------------|---------|-------------|
| **Major (X.0.0)** | Every 6-8 months | Breaking changes, major new features, architectural changes |
| **Minor (0.Y.0)** | Every 2 months | New features, non-breaking API additions, significant improvements |
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
- [ ] Trigger release workflows
- [ ] Generate release notes
- [ ] Announce in `#announcements` Discord channel

## Release Artifacts

Each release includes:

- **PyPI package**: `llama-stack` and `llama-stack-client`
- **npm package**: `llama-stack-client`
- **Docker images**: Distribution images on Docker Hub
- **GitHub Release**: Tagged release with release notes
- **Documentation**: Updated docs at https://llamastack.github.io

See [CONTRIBUTING.md](./CONTRIBUTING.md) for general contribution guidelines.

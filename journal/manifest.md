# Journal Manifest

**Project:** llama-stack
**Goal:** Contributing features and fixes to the llama-stack open source project
**Success Metrics:** CI passing + positive code reviews
**Reminder Cadence:** When patterns emerge
**Entry Format:** Mixed (adapt to content)
**Context Depth:** 3 journals
**Last Updated:** Wednesday February 18, 2026 14:28

## Domain-Specific Tracking
- CI/Pre-commit: Track CI failures, pre-commit hook issues, test patterns

## Pattern Categories
- Common mistakes to avoid
- Successful approaches
- Code patterns and conventions
- User preferences

---

## Patterns

### Good Patterns (Keep Doing)
- Triage review findings with user before acting — avoids re-doing settled decisions
- Use Pydantic `json_schema_extra` to modify OpenAPI output without changing Python types
- Run pre-commit hooks before pushing; stage hook-modified files and re-commit

### Anti-Patterns (Avoid)
- Re-reviewing code that was already iterated on multiple times — check history first
- Using `OSError` as a catch-all for network errors when `ConnectionError` + `TimeoutError` suffice

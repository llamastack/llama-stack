# Postmortem: 2025-01 Search Indexing Incident

**Date:** 2025-01-08 · **Duration:** 2 hours 15 minutes · **Severity:** SEV-2
**Author:** Anika Patel (Search Team) · **Status:** Action items complete

## Summary

On January 8, 2025 at 09:45 UTC, the product search service began returning
stale results for approximately 60% of queries.  New products added in the
previous 12 hours were not appearing in search, and price updates were not
reflected.  The incident lasted 2 hours 15 minutes until the indexing pipeline
was repaired at 12:00 UTC.

## Root Cause

**Elasticsearch bulk indexing failures caused by a mapping conflict after a
schema change was deployed without a corresponding index migration.**

The catalog team deployed a change that added a `variants` field (nested object
type) to the product schema.  However, the existing Elasticsearch index had
`variants` mapped as a `keyword` field from a previous prototype that was never
cleaned up.  The bulk indexer silently dropped documents that contained the new
nested `variants` structure, logging warnings but not raising alerts.

Over 12 hours, roughly 18,000 product documents failed to index.

## Timeline

| Time (UTC) | Event |
|------------|-------|
| 2025-01-07 21:30 | Catalog team deploys product schema change (adds nested `variants`) |
| 2025-01-07 21:32 | Bulk indexer begins logging `mapper_parsing_exception` warnings |
| 2025-01-08 09:45 | Customer support tickets spike — "new products not showing in search" |
| 2025-01-08 09:55 | Search team investigates; discovers indexing error logs |
| 2025-01-08 10:20 | Root cause identified: mapping conflict on `variants` field |
| 2025-01-08 10:45 | Fix: create new index with correct mapping, reindex from catalog DB |
| 2025-01-08 11:50 | Reindexing completes; alias swapped to new index |
| 2025-01-08 12:00 | Search results verified; incident resolved |

## Contributing Factors

1. **No schema migration for Elasticsearch** — The catalog team updated the
   application schema but did not run a corresponding ES index migration.  The
   deploy checklist did not include search index compatibility checks.
2. **Silent failures** — The bulk indexer logged warnings for mapping conflicts
   but did not alert or increment an error metric.  The warnings were lost in
   log noise.
3. **No freshness monitoring** — We had no alert for "time since last
   successful index update."  A 12-hour gap went unnoticed.
4. **Leftover prototype mapping** — The `variants` keyword field was added
   during a prototype 6 months ago and never removed.

## Action Items

| # | Action | Owner | Status |
|---|--------|-------|--------|
| 1 | Add ES mapping compatibility check to CI pipeline | Search Team | ✅ Done |
| 2 | Convert bulk indexer warnings to errors + PagerDuty alert | Search Team | ✅ Done |
| 3 | Add search freshness alert (warn if no docs indexed in 1 hour) | SRE | ✅ Done |
| 4 | Audit ES indices for stale/prototype mappings | Search Team | ✅ Done |
| 5 | Add index migration step to deploy checklist | Catalog Team | ✅ Done |

## Lessons Learned

- **Search indexes are part of the schema.** Changing the application data model
  without updating the search mapping is equivalent to skipping a database
  migration.
- **Silent drops are worse than loud failures.** The bulk indexer should have
  failed fast instead of silently skipping documents for 12 hours.
- **Monitor data freshness, not just availability.** The search service was "up"
  the entire time — it just served stale data.

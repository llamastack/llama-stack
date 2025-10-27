# Notion API Upgrade Analysis (v2025-09-03)

## Executive Summary

Notion released API version `2025-09-03` with **breaking changes** introducing first-class support for multi-source databases. This document analyzes the impact on the llama-stack project documentation system.

**Current Status:** ✅ No immediate action required
**Recommendation:** Monitor announcements, prepare migration plan, stay on v2022-06-28

---

## Current Configuration

### API Version in Use
```bash
NOTION_VERSION="2022-06-28"
NOTION_API_BASE="https://api.notion.com/v1"
```

### Databases
- **Llama Stack Database:** `299a94d48e1080f5bf20ef9b61b66daf`
- **Documentation Database:** `1fba94d48e1080709d4df69e9c0f0532`
- **Troubleshooting Database:** `1fda94d48e10804d843ee491d647b204`

### Primary Operations
- Creating pages in databases (`POST /v1/pages`)
- Publishing markdown documentation
- Registry tracking

---

## Breaking Changes Overview

### What Changed

Notion introduced **multi-source databases** - allowing a single database to contain multiple linked data sources. This fundamentally changes how pages are created and queried.

**Key Concept Change:**
```
OLD: One database = one data source (implicit)
NEW: One database = multiple data sources (explicit)
```

### When Changes Take Effect

**Immediately** upon upgrading to `2025-09-03`. No grace period or compatibility mode.

---

## Detailed Impact Analysis

### 1. Page Creation (CRITICAL)

**Current Method (v2022-06-28):**
```json
{
  "parent": {
    "database_id": "299a94d48e1080f5bf20ef9b61b66daf"
  },
  "properties": {...},
  "children": [...]
}
```

**New Method (v2025-09-03):**
```json
{
  "parent": {
    "data_source_id": "xxxx-xxxx-xxxx"  // Must fetch first!
  },
  "properties": {...},
  "children": [...]
}
```

**Migration Required:**
1. Fetch data source IDs for each database
2. Replace `database_id` with `data_source_id`
3. Update all JSON templates
4. Update upload scripts

### 2. Database Queries

**Current:**
```bash
POST /v1/databases/{database_id}/query
```

**New:**
```bash
POST /v1/data_sources/{data_source_id}/query
```

**Impact:** Query scripts need complete rewrite

### 3. Data Source ID Discovery

**New Required Step:**
```bash
# Must call before any operations
curl -X GET "https://api.notion.com/v1/databases/299a94d48e1080f5bf20ef9b61b66daf" \
  -H "Authorization: Bearer $NOTION_BEARER_TOKEN" \
  -H "Notion-Version: 2025-09-03"

# Response includes data_sources array
{
  "data_sources": [
    {"id": "actual-id-to-use-for-operations", ...}
  ]
}
```

### 4. Search API Changes

**Current:**
```json
{
  "filter": {
    "property": "object",
    "value": "database"
  }
}
```

**New:**
```json
{
  "filter": {
    "property": "object",
    "value": "data_source"  // Changed!
  }
}
```

### 5. Webhook Events

**Event Name Changes:**
```
database.created     → data_source.created
database.updated     → data_source.updated
database.deleted     → data_source.deleted
```

---

## Risk Assessment

### Low Risk Factors ✅
- We control when to upgrade (explicit version in API calls)
- Backward compatibility maintained for old versions
- Simple migration path (mostly find/replace)
- Limited scope (documentation publishing only)

### Medium Risk Factors ⚠️
- **No deprecation timeline announced** - could become urgent without warning
- **User-triggered failures** - if database owners add multi-source to our databases
- **Multiple databases to migrate** - 3+ databases to update

### High Risk Factors ❌
- None currently identified

---

## Migration Requirements

### Configuration Updates

**Add to .env:**
```bash
# Current
NOTION_VERSION="2022-06-28"

# After migration
NOTION_VERSION="2025-09-03"

# New variables needed
LLAMA_STACK_DATA_SOURCE_ID="[to-be-fetched]"
DOCS_DATA_SOURCE_ID="[to-be-fetched]"
TROUBLESHOOTING_DATA_SOURCE_ID="[to-be-fetched]"
```

### Script Updates

**Files requiring changes:**
1. `scripts/upload_notion_with_gpg.sh`
2. `docs/knowledge/upload_notion_validated.sh`
3. All JSON templates in `docs/knowledge/*-notion.json`

**Required modifications:**
- Add data source ID fetching logic
- Replace `database_id` with `data_source_id` in all API calls
- Update error handling for new response formats

### Documentation Updates

**Files to update:**
1. `docs/knowledge/notion-publishing-workflow.md`
2. `docs/knowledge/notion-collaboration-guide.md`
3. `docs/knowledge/doc_registry.md`
4. Project README sections on Notion integration

---

## Migration Plan

### Phase 1: Discovery (When Ready)

```bash
#!/bin/bash
# fetch_data_source_ids.sh

DATABASES=(
  "299a94d48e1080f5bf20ef9b61b66daf:LLAMA_STACK"
  "1fba94d48e1080709d4df69e9c0f0532:DOCS"
  "1fda94d48e10804d843ee491d647b204:TROUBLESHOOTING"
)

for db_info in "${DATABASES[@]}"; do
  db_id="${db_info%%:*}"
  db_name="${db_info##*:}"

  echo "Fetching data source for $db_name..."
  curl -X GET "https://api.notion.com/v1/databases/$db_id" \
    -H "Authorization: Bearer $NOTION_BEARER_TOKEN" \
    -H "Notion-Version: 2025-09-03" | \
    jq -r ".data_sources[0].id" > "/tmp/${db_name}_data_source_id.txt"
done
```

### Phase 2: Update Configuration

```bash
# Update .env with fetched IDs
LLAMA_STACK_DATA_SOURCE_ID=$(cat /tmp/LLAMA_STACK_data_source_id.txt)
DOCS_DATA_SOURCE_ID=$(cat /tmp/DOCS_data_source_id.txt)
TROUBLESHOOTING_DATA_SOURCE_ID=$(cat /tmp/TROUBLESHOOTING_data_source_id.txt)
```

### Phase 3: Update Scripts

```bash
# Find all JSON templates
find docs/knowledge -name "*-notion.json" -type f

# Update database_id to data_source_id
sed -i 's/"database_id":/"data_source_id":/g' docs/knowledge/*-notion.json

# Update shell scripts
# (Manual review and update required)
```

### Phase 4: Test & Validate

1. Create test page in development database
2. Verify page creation works
3. Test query operations
4. Validate search functionality
5. Check webhook events (if used)

### Phase 5: Production Migration

1. Backup current .env configuration
2. Apply all changes
3. Test with single document
4. Roll out to all operations
5. Update documentation

---

## Timeline & Recommendations

### Immediate (Now)
✅ Document analysis (this document)
✅ Monitor Notion changelog
✅ Create migration scripts (not execute)

### Short Term (3 Months)
- Stay on v2022-06-28
- No action required
- Continue monitoring

### Medium Term (When Announced)
- Execute migration when deprecation announced
- Or when multi-source features needed
- Or after 6+ months of stability

### Long Term
- Periodic reviews of Notion API changes
- Keep migration scripts updated
- Document all configuration changes

---

## Decision Matrix

### Stay on v2022-06-28 IF:
✅ Current version works without issues
✅ No deprecation timeline announced
✅ No need for multi-source features
✅ Prefer stability over new features

### Upgrade to v2025-09-03 IF:
- Deprecation announced for v2022-06-28
- Need multi-source database features
- Databases modified by owners (forced upgrade)
- 6+ months have passed (stability proven)

---

## Monitoring Strategy

### Quarterly Checks
1. Review Notion developer changelog
2. Check for deprecation announcements
3. Test current integration still works
4. Update migration scripts if needed

### Triggers for Immediate Action
🚨 Deprecation notice for v2022-06-28
🚨 Database owners add multi-source
🚨 Current version shows instability
🚨 Critical security fixes in new version

---

## Resources

### Official Documentation
- Upgrade Guide: https://developers.notion.com/docs/upgrade-guide-2025-09-03
- API Reference: https://developers.notion.com/reference
- Changelog: https://developers.notion.com/changelog

### Internal Documentation
- `docs/knowledge/notion-publishing-workflow.md`
- `docs/knowledge/notion-collaboration-guide.md`
- `docs/knowledge/doc_registry.md`

---

## Conclusion

**Current Recommendation:** **Do NOT upgrade yet**

**Rationale:**
- No immediate benefit
- No deprecation pressure
- Current system stable
- Migration effort not justified

**Next Review:** 3 months from now (or when Notion announces deprecation)

**Prepared By:** Claude Code
**Date:** October 2025
**Version:** 1.0

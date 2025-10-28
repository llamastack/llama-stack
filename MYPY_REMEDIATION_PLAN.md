# MyPy Error Remediation Plan

## Summary

After removing all mypy suppressions (except `strong_typing/auxiliary.py`), we have:
- **Total Errors**: 1,200 errors
- **Total Files**: 117 files
- **Suggested PRs**: 17 stacked PRs

## Top Error Types

| Error Type | Count | Description |
|------------|-------|-------------|
| union-attr | 324 | Accessing attributes on Optional/Union types without None checks |
| arg-type | 270 | Function argument type mismatches |
| assignment | 132 | Variable assignment type mismatches |
| attr-defined | 121 | Accessing undefined attributes on objects |
| no-any-return | 108 | Functions returning 'Any' instead of specific types |
| return-value | 36 | Return type doesn't match function signature |
| call-arg | 32 | Wrong arguments passed to function calls |
| var-annotated | 29 | Variables missing type annotations |
| override | 22 | Method override signature doesn't match parent |

## Stacked PR Plan

### HIGH Priority (Do First)

#### PR1: Core Routing Tables
- **Errors**: 163 | **Files**: 8 | **Complexity**: MEDIUM
- **Main errors**: union-attr(81), arg-type(42), no-any-return(25)
- **Description**: Fix routing tables (models, shields, vector_stores, common)
- **Top files**:
  - `core/routing_tables/common.py` (90 errors)
  - `core/routing_tables/models.py` (28 errors)
  - `core/routing_tables/vector_stores.py` (26 errors)
- **Why first**: Foundation for the routing system, affects other PRs

#### PR2: Core Routers
- **Errors**: 80 | **Files**: 7 | **Complexity**: MEDIUM
- **Main errors**: no-any-return(34), attr-defined(16), union-attr(12)
- **Description**: Fix core router implementations (inference, vector_io, safety)
- **Top files**:
  - `core/routers/inference.py` (29 errors)
  - `core/routers/vector_io.py` (28 errors)
  - `core/routers/safety.py` (9 errors)
- **Depends on**: PR1 (uses routing tables)

#### PR3: OpenAI Compatibility Layer
- **Errors**: 112 | **Files**: 2 | **Complexity**: HIGH
- **Main errors**: arg-type(37), union-attr(23), assignment(17)
- **Description**: Fix OpenAI/LiteLLM integration utils
- **Top files**:
  - `providers/utils/inference/openai_compat.py` (88 errors)
  - `providers/utils/inference/litellm_openai_mixin.py` (24 errors)
- **Independent**: Can be done in parallel with PR1-2

#### PR4: Meta Reference Agents
- **Errors**: 280 | **Files**: 9 | **Complexity**: HIGH
- **Main errors**: union-attr(128), arg-type(78), attr-defined(27)
- **Description**: Fix meta reference agent implementation
- **Top files**:
  - `providers/inline/agents/meta_reference/agent_instance.py` (81 errors)
  - `providers/inline/agents/meta_reference/responses/openai_responses.py` (76 errors)
  - `providers/inline/agents/meta_reference/responses/tool_executor.py` (45 errors)
- **Independent**: Can be done in parallel

---

### MEDIUM Priority (Do After HIGH)

#### PR5: Llama Models
- **Errors**: 123 | **Files**: 15 | **Complexity**: HIGH
- **Main errors**: assignment(31), attr-defined(26), no-any-return(13)
- **Description**: Fix llama model implementations (llama3, llama4, multimodal)
- **Top files**:
  - `models/llama/llama3/multimodal/model.py` (27 errors)
  - `models/llama/llama3/generation.py` (24 errors)

#### PR6: Provider Utils - Inference
- **Errors**: 41 | **Files**: 3 | **Complexity**: MEDIUM
- **Description**: Fix provider inference utilities (prompt_adapter, model_registry, embedding_mixin)

#### PR7: Provider Utils - Storage
- **Errors**: 36 | **Files**: 4 | **Complexity**: MEDIUM
- **Description**: Fix storage utilities (kvstore, memory, sqlstore)

#### PR8: Provider Utils - Other
- **Errors**: 28 | **Files**: 4 | **Complexity**: MEDIUM
- **Description**: Fix other provider utils (bedrock, telemetry, tools, scoring)

#### PR9: Inline Providers - Eval
- **Errors**: 53 | **Files**: 1 | **Complexity**: MEDIUM
- **Description**: Fix inline eval provider

#### PR10: Inline Providers - Safety
- **Errors**: 21 | **Files**: 4 | **Complexity**: MEDIUM
- **Description**: Fix inline safety providers (llama_guard, code_scanner)

#### PR12: Inline Providers - Inference & Post-Training
- **Errors**: 10 | **Files**: 2 | **Complexity**: HIGH
- **Description**: Fix inline inference and post-training providers

#### PR14: Remote Providers - Inference
- **Errors**: 67 | **Files**: 12 | **Complexity**: MEDIUM
- **Description**: Fix remote inference providers (bedrock, nvidia, tgi, together, etc.)

#### PR17: Core - Remaining
- **Errors**: 43 | **Files**: 7 | **Complexity**: MEDIUM
- **Description**: Fix remaining core modules (server, client, build, utils, store)

---

### LOW Priority (Can be parallelized or done last)

#### PR11: Inline Providers - Scoring
- **Errors**: 27 | **Files**: 10 | **Complexity**: MEDIUM
- **Description**: Fix inline scoring providers (basic, braintrust, llm_as_judge)

#### PR13: Inline Providers - Other
- **Errors**: 10 | **Files**: 2 | **Complexity**: LOW
- **Description**: Fix other inline providers (datasetio, vector_io)

#### PR15: Remote Providers - Vector IO
- **Errors**: 51 | **Files**: 11 | **Complexity**: MEDIUM
- **Description**: Fix remote vector_io providers (chroma, qdrant, weaviate, pgvector, milvus)

#### PR16: Remote Providers - Other
- **Errors**: 49 | **Files**: 15 | **Complexity**: MEDIUM
- **Description**: Fix other remote providers (safety, tool_runtime, post_training, etc.)

---

## Recommended Execution Strategy

### Phase 1: Foundation (Week 1)
1. **PR1**: Core Routing Tables (163 errors) - Start here
2. **PR2**: Core Routers (80 errors) - Stack on PR1

### Phase 2: High-Value Infrastructure (Week 1-2)
3. **PR3**: OpenAI Compatibility Layer (112 errors) - Parallel with Phase 1
4. **PR4**: Meta Reference Agents (280 errors) - Parallel with Phase 1

### Phase 3: Core Components (Week 2-3)
- **PR5**: Llama Models (123 errors)
- **PR17**: Core - Remaining (43 errors)
- Can be done in parallel

### Phase 4: Provider Utilities (Week 3-4)
- **PR6-10**: Provider Utils & Inline Providers
- Can be done in parallel by different developers

### Phase 5: Remote Providers (Week 4-5)
- **PR14-16**: Remote provider implementations
- Can be done in parallel

### Phase 6: Cleanup (Week 5)
- **PR11, PR13**: Low priority items

## Commands

```bash
# Run mypy with full type checking
uv run --group dev --group type_checking mypy

# Run mypy on specific module
uv run --group dev --group type_checking mypy src/llama_stack/core/routing_tables/

# See detailed errors for a file
uv run --group dev --group type_checking mypy src/llama_stack/core/routing_tables/common.py
```

## Notes

- All PRs should include tests where applicable
- Each PR should pass `uv run --group dev --group type_checking mypy` on its changed files
- Consider adding `--check-untyped-defs` flag for even stricter checking in the future
- Some errors may cascade - fixing one file might fix errors in dependent files

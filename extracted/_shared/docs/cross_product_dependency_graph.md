# Cross-Product Dependency Graph

This document tracks intended extracted-product dependencies. It is not a
runtime import map yet; it is the source of truth for extraction direction.

## Products

| Product | Current package | Intended dependencies |
|---|---|---|
| LLM Infrastructure | `extracted_llm_infrastructure` | Shared standalone substrate only |
| Competitive Intelligence | `extracted_competitive_intelligence` | LLM Infrastructure, host email/suppression adapters |
| Content Pipeline | `extracted_content_pipeline` | LLM Infrastructure, host publishing/storage adapters |
| Quality Gate | planned | LLM Infrastructure, Content Pipeline |
| Intent Router | planned | LLM Infrastructure, shared memory/tool registry |

## Current Notes

- Competitive Intelligence already uses extracted LLM substrate in standalone
  mode for protocols and LLM bridge wiring.
- Content Pipeline is active in PR #43; avoid physical package moves until that
  work lands.
- Shared scripts under `extracted/_shared/scripts/` are safe to use before the
  physical package consolidation.

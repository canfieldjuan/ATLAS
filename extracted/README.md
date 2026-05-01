# Extracted Products

This directory is the coordination layer for sellable systems being extracted
from Atlas. It does not replace the current `extracted_*` package directories
yet. Physical package moves should happen only after active product PRs are
merged and compatibility wrappers are planned.

## Current Products

| Product | Current path | Phase | Notes |
|---|---|---|---|
| LLM Infrastructure | `extracted_llm_infrastructure/` | Phase 2 | Standalone substrate landed; provider-level decoupling remains Phase 3. |
| Competitive Intelligence | `extracted_competitive_intelligence/` | Phase 2 | Standalone substrate landed; selected MCP read surfaces are product-owned. |
| Content Pipeline | `extracted_content_pipeline/` | Phase 2 in progress | Active PR work exists; do not physically move while PR #43 is active. |
| Quality Gate | planned | Not started | ProductClaim engine and frontend approval gate candidate. |
| Intent Router | planned | Not started | Semantic routing, memory quality, and tool registry candidate. |

## Layout Intent

The long-term layout is:

```text
extracted/
  _shared/
    scripts/
    docs/
    _standalone/
  competitive_intelligence/
  llm_infrastructure/
  content_pipeline/
  quality_gate/
  intent_router/
```

The current PR only adds the coordination layer and shared tooling. Existing
package import paths stay unchanged.

## Shared Tooling

Shared scripts live in `extracted/_shared/scripts/` and accept a product
directory such as `extracted_competitive_intelligence`.

```bash
python extracted/_shared/scripts/check_extracted_imports.py extracted_competitive_intelligence
bash extracted/_shared/scripts/validate_extracted.sh extracted_competitive_intelligence
bash extracted/_shared/scripts/check_ascii_python.sh extracted_competitive_intelligence
```

Use `--no-atlas-fallback` with `check_extracted_imports.py` for products that
have moved to strict extracted-only relative import resolution. Products still
in early scaffold mode can omit the flag to match the current LLM-infrastructure
transition behavior.

Existing per-product scripts remain the CI entry points until a later migration
switches them to call the shared scripts.

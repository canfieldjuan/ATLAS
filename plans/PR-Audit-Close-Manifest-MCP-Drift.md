# PR-Audit-Close-Manifest-MCP-Drift

## Why this slice exists

Two useful auditors were intentionally left out of the pre-push wrapper
because they exposed known drift: extracted manifest sync drift and
CLAUDE.md MCP tool-name inventory drift. This slice closes those drifts
and wires both auditors into the local/GitHub mechanical review path.

## Scope (this PR)

1. Fix extracted manifest drift for content pipeline, LLM infrastructure,
   and quality gate mapped files.
2. Fix CLAUDE.md MCP tool-name inventory drift.
3. Add the now-green manifest and tool-name auditors to
   `scripts/pre_push_audit.sh`.
4. Refresh the in-flight coordination row for this slice.

### Files touched

- `CLAUDE.md`
- `atlas_brain/services/b2b/product_claim.py`
- `extracted_content_pipeline/manifest.json`
- `extracted_content_pipeline/services/b2b/vendor_briefing_delivery.py`
- `extracted_llm_infrastructure/services/b2b/llm_exact_cache.py`
- `extracted_quality_gate/product_claim.py`
- `scripts/pre_push_audit.sh`
- `plans/PR-Audit-Close-Manifest-MCP-Drift.md`
- `docs/extraction/coordination/inflight.md`

## Mechanism

For extracted manifests:

- `vendor_briefing_delivery.py` was listed as both mapped and owned in
  `extracted_content_pipeline/manifest.json`. The owned duplicate was
  removed so the mapped source is canonical, then the extracted target was
  synced from Atlas.
- `llm_exact_cache.py` was synced from Atlas into
  `extracted_llm_infrastructure`, bringing the LLM Gateway namespace gate
  into the extracted package.
- `product_claim.py` now keeps the standalone-safe `StrEnum` fallback in
  the Atlas source, then syncs the quality-gate target from that source.

For MCP tool inventory drift:

- `CLAUDE.md` now includes Email `list_folders`.
- Calendar and Memory prose no longer backticks non-tool table names.
- B2B Churn Intelligence now documents the 22 previously missing tools.

After both auditors pass, `scripts/pre_push_audit.sh` runs them on every
local review and GitHub pre-push-audit workflow.

## Intentional

- ProductClaim's module docstring is changed to standalone-core language so
  the Atlas source and extracted quality-gate target can stay byte-identical.
- The content-pipeline delivery helper is mapped-only now; the previous
  mapped-plus-owned manifest state contradicted the sync contract.
- This does not touch the deferred review-source count auditor.

## Deferred

- Optional local hook installer remains the next slice.
- Shell hygiene auditor remains deferred from the previous audit-kit work.

## Verification

```bash
python scripts/audit_extracted_manifests.py
python scripts/audit_mcp_tool_names_match_docs.py
bash scripts/local_pr_review.sh
python scripts/audit_plan_doc.py plans/PR-Audit-Close-Manifest-MCP-Drift.md
python scripts/audit_plan_doc_files_touched.py plans/PR-Audit-Close-Manifest-MCP-Drift.md origin/main
python scripts/audit_plan_doc_diff_size.py plans/PR-Audit-Close-Manifest-MCP-Drift.md origin/main
bash -n scripts/pre_push_audit.sh
python -m py_compile atlas_brain/services/b2b/product_claim.py extracted_quality_gate/product_claim.py extracted_llm_infrastructure/services/b2b/llm_exact_cache.py
git diff --check
```

## Estimated diff size

| File | LOC churn (approx) |
|---|---:|
| `CLAUDE.md` | 24 |
| `atlas_brain/services/b2b/product_claim.py` | 23 |
| `docs/extraction/coordination/inflight.md` | 4 |
| `extracted_content_pipeline/manifest.json` | 3 |
| `extracted_content_pipeline/services/b2b/vendor_briefing_delivery.py` | 2 |
| `extracted_llm_infrastructure/services/b2b/llm_exact_cache.py` | 33 |
| `extracted_quality_gate/product_claim.py` | 2 |
| `scripts/pre_push_audit.sh` | 2 |
| `plans/PR-Audit-Close-Manifest-MCP-Drift.md` | 94 |
| **Total** | **~187** |

# PR: Content Ops Persisted Source Execute

## Why this slice exists

PR #1228 closed the immediate uploaded CSV handoff by letting the New Run UI
apply full normalized rows to `inputs.source_material`. Its deferred backend
item remains: when rows have already been imported into
`campaign_opportunities`, execute should be able to load those persisted target
IDs by tenant scope instead of requiring the operator to inline the full row
payload again.

This slice adds the smallest persisted-source execution path behind the
existing Atlas input-provider seam. It lands slightly over the 400 LOC soft
budget because the tenant-scope guard, ambiguity guard, missing-row warning, and
route-level execute proof all need to ship with the new persisted lookup path;
dropping any one of those would repeat the false-green shape from #1228.

## Scope (this PR)

Ownership lane: content-ops/upload-source-run-handoff

Slice phase: Vertical slice

1. Teach the Atlas Content Ops input provider to accept persisted import target
   IDs from request inputs.
2. Resolve those IDs through the existing tenant-scoped
   `PostgresIntelligenceRepository.read_campaign_opportunities(...)`.
3. Fail closed when no tenant account scope or no DB provider is available:
   report warnings and do not load unscoped/global rows.
4. Reuse `SupportTicketInputProvider` to convert loaded rows into the existing
   FAQ/landing/blog `source_material` package.
5. Add host-provider and route tests proving tenant-scoped lookup, missing-row
   warnings, no-scope skip, and execute using persisted source rows.

### Files touched

- `atlas_brain/_content_ops_input_provider.py`
- `.github/workflows/atlas_content_ops_input_provider_checks.yml`
- `tests/test_atlas_content_ops_input_provider.py`
- `plans/PR-Content-Ops-Persisted-Source-Execute.md`

## Mechanism

The request contract is intentionally small and provider-owned. The input
provider recognizes these aliases under `inputs`:

```json
{
  "source_import_target_ids": ["ticket-1", "ticket-2"]
}
```

It also accepts `source_target_ids` and `import_target_ids` for compatibility
with operator JSON. The provider deduplicates non-empty IDs, requires
`scope.account_id`, then reads each ID with:

```python
repository.read_campaign_opportunities(
    scope=scope,
    target_mode=request.target_mode or "vendor_retention",
    limit=2,
    filters={"target_id": target_id},
)
```

Exactly one returned row is loaded; zero rows are reported missing and more
than one row is reported ambiguous so lookup fails closed instead of guessing.
Loaded rows are combined with any inline `source_material` or selected FAQ
outputs and handed to `SupportTicketInputProvider`. Missing IDs and skipped
unscoped lookups are surfaced as input-provider warnings. No generation
services or import SQL are reimplemented.

## Intentional

- No UI change in this slice. The backend execution contract lands first; the
  next UI slice can choose whether to write target IDs instead of full
  `source_material`.
- No new database adapter. The existing `PostgresIntelligenceRepository`
  already handles table identifiers, row normalization, and account filtering
  when scoped.
- No unscoped fallback. If `scope.account_id` is empty, persisted target IDs
  are skipped and no repository call is made.
- Reads are one target ID at a time. Request input arrays are already bounded,
  and this keeps the slice within the current repository API.

## Deferred

- Future PR: New Run UI control that applies imported `targetIds` as
  `source_import_target_ids` instead of applying full `source_material`.
- Future PR: batch `target_id IN (...)` repository helper if persisted source
  selection needs larger operator-managed batches.
- Future PR: browser E2E against a live uploaded support-ticket CSV producing
  an approved public landing/blog asset.
- Parked hardening: none. `HARDENING.md` and `ATLAS-HARDENING.md` were scanned;
  existing entries are dependency audit and blog content-quality issues, not
  this persisted-source execution path.

## Verification

- `python -m py_compile atlas_brain/_content_ops_input_provider.py
  tests/test_atlas_content_ops_input_provider.py` - passed.
- `pytest tests/test_atlas_content_ops_input_provider.py -k
  'persisted or source_import_targets_unscoped or
  source_import_targets_ambiguous' -q` - 4 passed, 19 deselected.
- `pytest tests/test_atlas_content_ops_input_provider.py -q` - 23 passed, 1
  warning from the existing optional `atlas_brain.api` import path.
- `python scripts/audit_extracted_pipeline_ci_enrollment.py
  --atlas-brain-tests-from origin/main` - passed.
- `bash scripts/local_pr_review.sh --current-pr-body-file
  /tmp/content-ops-persisted-source-execute-pr-body.md` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~100 |
| Dedicated Atlas workflow enrollment | ~40 |
| Host input provider | ~160 |
| Host/provider and execute route tests | ~220 |
| **Total** | **~520** |

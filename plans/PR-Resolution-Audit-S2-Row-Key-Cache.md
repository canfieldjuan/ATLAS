# PR-Resolution-Audit-S2-Row-Key-Cache

## Why this slice exists

Issue #1993 S2 tracks a live current-code performance defect in the
Resolution Audit CSV path. The finding is confirmed against current code, not
accepted from the older audit text: `campaign_source_adapters._SourceFieldLookup`
already shows a per-row cached alias lookup pattern, while
`support_ticket_input_package._first_value` and
`ticket_faq_markdown._field_value` still normalize raw row keys while scanning
`row.items()` for every field lookup.

Root cause: the support-ticket input package and FAQ markdown builder do not
carry a cached field lookup object through their repeated per-row lookup hot
paths. They re-run the same raw-key normalization work for each alias lookup
even though the row is unchanged.

Correct fix contract:

1. Cache normalized row-key lookups once per row in the support-ticket input
   package while preserving current alias order, blank-value skipping, and
   exact `row.get(...)` passthrough semantics.
2. Cache source-type/compact row-key lookups once per opportunity/evidence row
   in the FAQ markdown builder while preserving exact-key precedence and output
   values.
3. Prove parity with realistic fixtures so rendered/package outputs do not
   change.
4. Prove the hot path no longer repeatedly scans raw row keys for cached
   lookups.

This fixes the root for S2's queued defect; it is not a user-facing product
shape change.

Diff-budget note: this slice is slightly over 400 LOC because the root fix
requires two module-local cache wrappers plus parity/cache guards in both
existing focused test files. Splitting the wrappers from their proof would
leave the hot-path claim unprotected.

## Scope (this PR)

Ownership lane: resolution-audit-csv
Slice phase: Functional validation

1. Add local per-row lookup wrappers for the support-ticket package and FAQ
   markdown builder lookup helpers.
2. Route the existing normalization/build hot paths through those wrappers.
3. Add parity tests and focused cache-use guards.
4. Update the current-code remediation tracker for S2.

### Review Contract

- Acceptance criteria:
  - Support-ticket package outputs remain identical for alias-heavy rows,
    including blank source-ID fallback behavior from S1.
  - FAQ markdown outputs remain identical for rows whose fields are reachable
    only through compact/source-type alias matching.
  - Cache guards show repeated lookups do not rescan the raw mapping.
  - `docs/audits/resolution-audit-csv/CURRENT_CODE_REMEDIATION_ARC.md` records
    S2 as addressed by this PR.
- Affected surfaces:
  - `extracted_content_pipeline/support_ticket_input_package.py`
  - `extracted_content_pipeline/ticket_faq_markdown.py`
  - focused tests for those modules
  - Resolution Audit CSV remediation tracker
- Risk areas:
  - Alias precedence and blank-value semantics in support-ticket rows.
  - Exact-key precedence in FAQ markdown rows.
  - Accidentally widening fuzzy lookup behavior to ordinary `row.get(...)`.
- Reviewer rules triggered: R1, R10, R13, R14.

### Files touched

- `docs/audits/resolution-audit-csv/CURRENT_CODE_REMEDIATION_ARC.md`
- `extracted_content_pipeline/support_ticket_input_package.py`
- `extracted_content_pipeline/ticket_faq_markdown.py`
- `plans/PR-Resolution-Audit-S2-Row-Key-Cache.md`
- `tests/test_extracted_support_ticket_input_package.py`
- `tests/test_extracted_ticket_faq_markdown.py`

## Mechanism

The support-ticket path gets a mapping wrapper that delegates normal mapping
operations to the raw row, but indexes compact normalized raw keys once. The
existing `_first_value` and `_has_any_key` helpers use the wrapper when present,
so direct passthrough `row.get(...)` remains exact and unchanged.

The FAQ markdown path gets the same shape for `_field_value`: exact keys still
win first, then cached source-type/compact aliases are consulted without
rescanning `row.items()`. `build_ticket_faq_markdown` wraps each opportunity
and evidence row once before calling the existing helper graph.

Tests cover both the semantic parity and the cache behavior. The cache guards
use counting mappings to prove repeated helper calls do not re-enter the raw
row-item scan after the wrapper is built.

## Intentional

- No shared cross-module abstraction in this slice. The lookup rules are close
  but not identical: support-ticket lookups skip blank values, while FAQ
  markdown preserves blank/falsey values and exact-key precedence.
- No user-facing report, snapshot, landing, email, or PDF shape changes. S2 is
  an internal hot-path remediation.
- No clustering, source-ID, or admission policy changes. Those are separate
  queued items in #1993.

## Deferred

- Broader runtime benchmarking across large customer-style uploads remains
  deferred; this slice adds a focused regression guard for the repeated scan
  class without turning the PR into a benchmark harness.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_extracted_support_ticket_input_package.py -q`
  (90 passed)
- `python -m pytest tests/test_extracted_ticket_faq_markdown.py -q`
  (424 passed)
- `scripts/maturity_sweep_file_lane.py` on `extracted_content_pipeline/ticket_faq_markdown.py`
  with `tests/maturity_sweep/baseline_extracted_content_pipeline.json`
  (ratchet passed; score back to baseline 16)
- `scripts/maturity_sweep.py` on `extracted_content_pipeline` with
  `tests/maturity_sweep/baseline_extracted_content_pipeline.json`
  (ratchet passed)
- `scripts/validate_extracted_content_pipeline.sh` via bash (passed)
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline`
  (passed)
- `python scripts/audit_extracted_standalone.py --fail-on-debt` (passed)
- `scripts/check_ascii_python.sh` via bash (passed)

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/audits/resolution-audit-csv/CURRENT_CODE_REMEDIATION_ARC.md` | 12 |
| `extracted_content_pipeline/support_ticket_input_package.py` | 52 |
| `extracted_content_pipeline/ticket_faq_markdown.py` | 60 |
| `plans/PR-Resolution-Audit-S2-Row-Key-Cache.md` | 142 |
| `tests/test_extracted_support_ticket_input_package.py` | 120 |
| `tests/test_extracted_ticket_faq_markdown.py` | 118 |
| **Total** | **504** |

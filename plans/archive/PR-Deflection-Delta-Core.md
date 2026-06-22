# PR-Deflection-Delta-Core

## Why this slice exists

#1316 defines the monthly delta report as the subscription retention engine,
but it was blocked until #1768 added stable paid action-row identity. That
identity foundation is now merged, so the next safe slice is the pure D1 core:
compare two persisted `deflection.v1` report models without touching storage,
delivery, rendering, or the paid flow.

Root cause: the codebase has paid action rows with stable identity and business
signals, but no pure comparator that turns two report models into auditable
change rows. This PR fixes that root for the non-I/O core only; persistence,
baseline selection, and customer-facing delta delivery remain separate slices.

Diff-size note: this runs over the 400 LOC target because the smallest useful
D1 slice needs the pure comparator, focused negative/edge tests, manifest
ownership, and CI runner enrollment in the same PR. Splitting the test
enrollment or low-confidence proof would leave the core under-protected.

## Scope (this PR)

Ownership lane: issue-1316/deflection-delta-core
Slice phase: Vertical slice

1. Add a pure `compute_deflection_delta(current_model, baseline_model)` helper
   for supported `deflection.v1` report models.
2. Use paid action-row identity from #1768 (`repeat_key`/`cluster_id`) rather
   than rank, question ID, or raw question text.
3. Classify row-level changes for `NEW`, `RESOLVED`, `GROWING`, `SHRINKING`,
   `STILL_UNRESOLVED`, `STATUS_CHANGED`, `COST_CHANGED`, `CSAT_CHANGED`, and
   `RESURFACED` when the current row is already-covered-but-recurring.
4. Return a JSON-serializable `deflection_delta.v1` dict with aggregate counts
   and support-cost deltas.
5. Add focused pure tests with no DB/network.

### Review Contract

Acceptance criteria:
- The delta core compares by stable identity fields only; rank and `question_id`
  changes must not create delete/add pairs.
- New and resolved rows are detected when identity appears on only one side.
- Ticket-count, support-cost, status, and CSAT changes produce explicit
  change types.
- Low-confidence or missing identity is surfaced as an unmatched/low-confidence
  row rather than silently merged.
- Duplicate identity keys on either side are treated as ambiguous for both
  sides, so the comparator cannot silently choose one arbitrary row.
- Malformed numeric count/cost metadata is rejected instead of being coerced
  into a false zero delta.
- The result is deterministic and JSON-serializable.

Affected surfaces:
- `.github/workflows/extracted_pipeline_checks.yml`
- `extracted_content_pipeline/deflection_delta.py`
- `extracted_content_pipeline/manifest.json`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_content_ops_deflection_delta.py`

Risk areas:
- Lying about monthly change by matching on rank or display text.
- Pretending two-report comparison can solve all future fuzzy merge/split
  cases.
- Over-reaching into persistence or customer delivery before the core is
  proven.

Reviewer rules triggered: R1, R2, R10, R13, R14.

### Files touched

- `.github/workflows/extracted_pipeline_checks.yml`
- `extracted_content_pipeline/deflection_delta.py`
- `extracted_content_pipeline/manifest.json`
- `plans/PR-Deflection-Delta-Core.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_content_ops_deflection_delta.py`

## Mechanism

The helper extracts action rows from the paid model's `backlog_table` section
because that is the broadest bounded paid action-row set. If `backlog_table` is
absent, it falls back to `priority_fix_queue` so older in-flight model fixtures
can still be compared explicitly.

Rows are keyed by `repeat_key` first, then `cluster_id`; low-confidence,
missing, or duplicate keys are not merged with any other row. Duplicate keys
are counted before indexing, and any key that is non-unique in either model is
demoted to unmatched rows on both sides. For each matched key, the core compares
status, ticket count, estimated support cost, and CSAT signal fields. It emits
one change row per identity with a `change_types` list and before/after values
needed by later persistence/read-surface slices.

## Intentional

- No database, access-layer, API, MCP, email, PDF, or result-page changes in
  this slice.
- No fuzzy/merge/split matching yet. Exact stable identity must work first, and
  low-confidence matches are called out instead of guessed.
- Duplicate identities are treated as ambiguous data, not fuzzy matches. A
  later fuzzy/split/merge slice can choose a smarter strategy with real
  evidence.
- `RESURFACED` is limited to the current model's
  `Already covered but still recurring` status in this two-report core.
  Multi-period "was gone, came back" history belongs with persisted baseline
  selection.
- `backlog_table` is the canonical D1 row source because it covers the broader
  bounded paid backlog; narrower sections are later presentation views.

## Deferred

- #1316 D2 persistence and baseline selection.
- #1316 D3 paid-gated API/MCP read surface.
- #1316 D4 monthly delivery/upsell email.
- Fuzzy/merge/split identity matching and multi-period resurfaced detection.

Parked hardening: none.

## Verification

- Focused delta-core pytest -- 8 passed.
- Extracted CI enrollment audit -- 189 matching tests enrolled.
- Python compile for touched Python files -- passed.
- Git whitespace check -- passed.
- Broad extracted-content maturity ratchet -- passed.
- Deflection content-ops maturity ratchet -- passed.
- Extracted content pipeline validation -- passed.
- Extracted reasoning-import guard -- clean.
- Extracted standalone audit -- Atlas runtime import findings: 0.
- ASCII Python policy check -- passed.
- Pending before push: local PR review.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/extracted_pipeline_checks.yml` | 2 |
| `extracted_content_pipeline/deflection_delta.py` | 282 |
| `extracted_content_pipeline/manifest.json` | 3 |
| `plans/PR-Deflection-Delta-Core.md` | 142 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_content_ops_deflection_delta.py` | 237 |
| **Total** | **667** |

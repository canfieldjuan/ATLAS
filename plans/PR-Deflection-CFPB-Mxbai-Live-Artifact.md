# PR-Deflection-CFPB-Mxbai-Live-Artifact

## Why this slice exists

Issue #1504 now has the exact lexical join, embedding-booster core, host
adapter, and explicit CFPB compare harness merged. The remaining proof gap is
the operator-approved live CFPB re-baseline: run real CFPB source rows through
the same harness with the pinned offline mxbai host adapter, then attach a
sanitized artifact so the issue records whether the relative defaults are still
acceptable on the live source path.

This slice is validation, not a tuning slice. If the live run shows the current
mutual-nearest-neighbor margin or loose floor is wrong, this PR records that as
the outcome and defers threshold changes to a scoped follow-up.

## Scope (this PR)

Ownership lane: deflection/clustering
Slice phase: Functional validation

1. Add explicit embedding-probe telemetry to the CFPB compare smoke payload so
   live artifacts can prove the host embedding path ran.
2. Add focused regression coverage for successful probe telemetry, swallowed
   embedding exceptions, and invalid embedding batches.
3. Add a dated validation note for the live CFPB compare run enabled by #1542.
4. Commit a sanitized summary artifact showing baseline vs boosted counts,
   delta, output-check status, source-profile counts, and whether the embedding
   port actually produced a valid batch.
5. Leave raw CFPB JSONL rows and full generated Markdown in `tmp/` only.
6. Update #1504 with the run result and remaining backlog after the PR opens.

### Files touched

- `docs/extraction/validation/deflection_cfpb_mxbai_live_artifact_2026-06-14.md`
- `docs/extraction/validation/fixtures/deflection_cfpb_mxbai_live_artifact_20260614/summary.json`
- `plans/PR-Deflection-CFPB-Mxbai-Live-Artifact.md`
- `scripts/smoke_content_ops_cfpb_faq_markdown.py`
- `tests/test_smoke_content_ops_cfpb_faq_markdown.py`

### Review Contract

- Acceptance criteria:
  - [ ] The live command uses `--compare-embedding-booster` so baseline and
        boosted results run from the same fetched CFPB rows.
  - [ ] The committed artifact proves the run exited successfully, records a
        successful embedding comparison, and includes no raw CFPB source rows or
        full complaint narratives.
  - [ ] The validation note interprets the baseline/boosted delta instead of
        silently treating threshold movement as approved.
  - [ ] No production route flag, default, threshold, or runtime behavior changes
        in this PR.
- Affected surfaces: CFPB smoke script, focused smoke tests, docs, and committed
  validation artifact.
- Risk areas: misleading validation evidence, accidental raw-data commit,
  stale issue state.
- Reviewer rules triggered: R1, R2, R5, R10, R11, R12, R14.

## Mechanism

Extend `scripts/smoke_content_ops_cfpb_faq_markdown.py` so compare-mode payloads
include the embedding probe's call and valid-batch counts. The #1542 harness
already fails closed if the host embedding adapter cannot be constructed,
inference raises, the returned batch shape is invalid, or no valid embedding
batch is applied; this slice makes that proof visible in the emitted JSON.

Run the smoke script against live CFPB source rows with
`--compare-embedding-booster`, writing raw working files under
`tmp/cfpb_mxbai_live_artifact/`.

After the run, distill the JSON payload into a small committed summary under
`docs/extraction/validation/fixtures/`. The summary keeps counts, deltas,
source-profile metadata, output checks, and the selected question labels, but
does not commit raw source rows or full generated Markdown.

## Intentional

- No threshold, clustering, route, or service behavior changes. The only code
  change is smoke-harness telemetry needed for reviewable validation evidence.
- No production enablement. The hosted route flag remains a separate operator
  decision after the live artifact is reviewed.
- The committed artifact may include generated question labels because they are
  the summarized report surface being validated; raw CFPB rows, evidence blocks,
  and full complaint narratives stay out of git.

## Deferred

- Threshold or margin changes, if the live run proves the defaults are wrong.
- Semantic merge provenance surfacing for buyer-facing inspection, still tracked
  from the embedding-booster follow-ups.
- Safer multi-row lexical component expansion from #1536 if future live data
  needs semantic expansion beyond singleton lexical components.

Parked hardening: none.

## Verification

- Live CFPB compare command documented in
  `docs/extraction/validation/deflection_cfpb_mxbai_live_artifact_2026-06-14.md`
  -- passed, exit 0; embedding probe recorded 14 calls and 14 valid batches.
- Focused CFPB smoke tests for
  `tests/test_smoke_content_ops_cfpb_faq_markdown.py` -- 8 passed.
- Extracted pipeline check via `scripts/run_extracted_pipeline_checks.sh` --
  4116 passed, 10 skipped, 1 warning.
- Local PR review bundle with the planned PR body file -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/extraction/validation/deflection_cfpb_mxbai_live_artifact_2026-06-14.md` | 74 |
| `docs/extraction/validation/fixtures/deflection_cfpb_mxbai_live_artifact_20260614/summary.json` | 156 |
| `plans/PR-Deflection-CFPB-Mxbai-Live-Artifact.md` | 115 |
| `scripts/smoke_content_ops_cfpb_faq_markdown.py` | 4 |
| `tests/test_smoke_content_ops_cfpb_faq_markdown.py` | 48 |
| **Total** | **397** |

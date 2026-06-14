# PR-Deflection-CFPB-Embedding-Baseline

## Why this slice exists

Issue #1504 has the lexical repeat fix, the deterministic embedding booster,
and the host mxbai adapter merged, but the public CFPB smoke still proves only
the no-port FAQ path. Operators need one repeatable command that fetches the
same live CFPB rows once, runs the baseline and embedding-boosted FAQ paths
side by side, and emits a small JSON comparison before we spend time on a
larger live artifact.

Post-review note: Codex P2 found that compare mode could still report the
boosted path as primary when the host port was constructed but the actual
embedding inference call was swallowed inside the FAQ builder. The fix is
required before this harness can be trusted for a live artifact run; that pushes
the PR over the 400 LOC soft target, but splitting it would leave the review
finding open on the only command meant to validate mxbai live.

## Scope (this PR)

Ownership lane: deflection/clustering
Slice phase: Vertical slice

1. Add an explicit CFPB FAQ smoke compare mode that runs the no-embedding
   baseline and the embedding-boosted path from the same fetched rows.
2. Build the embedding port lazily from the host adapter only when compare mode
   is requested, so the default offline smoke behavior stays unchanged.
3. Add JSON summary fields for baseline, boosted, and delta results so a live
   run can show whether mxbai changes generated FAQ items, repeated-ticket
   accounting, and top questions.
4. Add focused tests that use a fake embedding port and mocked CFPB fetches; CI
   must not load the real model or call CFPB.
5. Fail compare mode if no valid embedding batch is applied, including the
   inference-time failure case the FAQ builder otherwise swallows.
6. Do not touch `scripts/smoke_content_ops_deflection_submit_handoff.py` or
   `tests/test_smoke_content_ops_deflection_submit_handoff.py`; those belong to
   open PR #1541.

### Files touched

- `extracted_content_pipeline/README.md`
- `plans/PR-Deflection-CFPB-Embedding-Baseline.md`
- `scripts/smoke_content_ops_cfpb_faq_markdown.py`
- `tests/test_smoke_content_ops_cfpb_faq_markdown.py`

### Review Contract

- Acceptance criteria:
  - [ ] Existing CFPB FAQ smoke output remains compatible when compare mode is
        not requested.
  - [ ] Compare mode fetches source rows once, runs baseline and boosted FAQ
        generation from the same loaded opportunities, and returns both result
        summaries plus a deterministic delta.
  - [ ] Compare mode fails closed with a clear error if the host embedding port
        cannot be built.
  - [ ] Compare mode fails closed with a clear error if the host embedding port
        is built but inference fails or returns no valid batch.
  - [ ] Tests prove the compare path uses an injected embedding port without
        loading the real model or making a network call.
- Affected surfaces: CLI smoke script, extracted pipeline tests, operator docs.
- Risk areas: backcompat, CI enrollment, dependency/config boundary,
  offline-vs-live behavior.
- Reviewer rules triggered: R1, R2, R5, R10, R11, R12, R14.

## Mechanism

The CFPB FAQ smoke will keep its current default path. A new explicit compare
flag will call the existing FAQ builder twice after one fetch and one source-row
load: first without an embedding port, then with an embedding port. The default
port factory will lazily import the host adapter so importing the smoke module
and running the baseline path do not require host embedding dependencies.

The JSON payload will keep the existing `faq` key for the primary rendered
result and add an `embedding_comparison` object with compact summaries for the
baseline and boosted runs plus a delta. Tests will inject a fake port factory
and assert fetch count, result shape, failure behavior, and unchanged default
behavior.

Compare mode wraps the embedding port in a small probe before passing it into
the FAQ builder. The probe records whether `embed_texts` was called, whether a
same-length non-string batch came back, and whether inference raised. After the
boosted build returns, the smoke treats any probe error or missing valid batch
as a failed compare and falls back to the baseline result instead of reporting a
false boosted primary.

## Intentional

- No live CFPB request or real mxbai model load in CI; the committed proof is
  the deterministic harness behavior, and the operator live run remains
  explicit.
- No production feature flag changes; this is an operator smoke command.
- No hosted submit smoke changes; #1541 owns that surface.
- No semantic merge provenance in generated FAQ Markdown yet; this slice only
  exposes the baseline-vs-boosted comparison needed to guide the next artifact
  run.

## Deferred

- Run the operator-approved live CFPB baseline comparison after this PR lands
  and attach the sanitized summary to #1504.
- Add semantic merge provenance to the report/UI path once the comparison
  proves the booster is worth surfacing.
- Revisit safer multi-row semantic component expansion after the live delta is
  understood.

Parked hardening: none.

## Verification

- Focused CFPB FAQ smoke pytest command: 7 passed.
- Extracted pipeline checks command: 4112 passed, 10 skipped; extracted
  reasoning core 295 passed.
- Local PR review command: passed on the committed diff.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/README.md` | 19 |
| `plans/PR-Deflection-CFPB-Embedding-Baseline.md` | 123 |
| `scripts/smoke_content_ops_cfpb_faq_markdown.py` | 147 |
| `tests/test_smoke_content_ops_cfpb_faq_markdown.py` | 136 |
| **Total** | **425** |

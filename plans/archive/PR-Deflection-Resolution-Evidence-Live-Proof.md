# PR-Deflection-Resolution-Evidence-Live-Proof

## Why this slice exists

Issue #1419's remaining gap is the publishable-answer lane: #1424/#1428 now
surface and assert the resolution-evidence signal, while #1408's real-data
proof showed CFPB complaint data correctly stays in the "no proven answer yet"
lane because it has no agent resolutions. The unresolved launch question is
whether a realistic provider export with actual support resolutions produces a
paid deflection report with `drafted_answer_count > 0`.

This slice creates that missing proof without changing generation semantics:
use a representative CSV shaped like a help-desk export, include resolved and
question-only rows, generate the deterministic FAQ deflection report, commit
the generated report/summary/result artifacts, and pin the proof with focused
tests.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Functional validation

1. Add a resolution-bearing support-ticket export fixture with realistic
   help-desk fields and both resolved and unresolved repeat-question themes.
2. Generate and commit the deflection report proof artifacts from that fixture:
   source CSV, summary JSON, result JSON, and report Markdown.
3. Add focused tests that load the committed fixture/artifacts and assert the
   publishable-answer lane (`drafted_answer_count > 0`,
   `support_ticket_resolution_evidence_present == true`) while preserving at
   least one `draft_needs_review` item.
4. Archive the already-merged #1428 plan as same-session teardown housekeeping.
5. Do not touch PDF/email/delivery worker code or paid delivery logic.

### Review Contract

- Acceptance criteria:
  - [ ] The committed source fixture has at least one repeated theme with
        scoped `resolution_text` and at least one repeated theme without
        resolution evidence.
  - [ ] The generated summary proves `drafted_answer_count > 0`,
        `support_ticket_resolution_evidence_present == true`, and
        `no_proven_answer_count > 0`.
  - [ ] The generated report Markdown contains the "Publishable Help-Center
        Copy From Proven Resolutions" section with real resolved steps and the
        "No Proven Answer Yet" section for the unresolved lane.
  - [ ] Tests would fail if the fixture becomes question-only or if the report
        collapses every item into the publishable-answer lane.
  - [ ] No live model route or local Ollama route is involved; FAQ deflection
        report generation remains deterministic.
- Affected surfaces: committed validation fixture/artifacts, deterministic
  deflection report CLI proof, focused proof tests, and merged-plan archive
  housekeeping.
- Risk areas: false-green proof artifacts, hand-crafted shapes that do not
  match real producer output, accidentally proving only the happy path, and
  drift into the PDF/delivery lane.
- Reviewer rules triggered: R1, R2, R10, R13.

### Files touched

- `docs/extraction/validation/deflection_resolution_evidence_live_proof_2026-06-09.md`
- `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/report.md`
- `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/result.json`
- `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/source.csv`
- `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/summary.json`
- `plans/INDEX.md`
- `plans/PR-Deflection-Resolution-Evidence-Live-Proof.md`
- `plans/archive/PR-Deflection-Resolution-Evidence-Absent-Assertions.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_content_ops_deflection_resolution_live_proof.py`

## Mechanism

The existing deterministic report CLI already runs the path this proof needs:

```bash
python scripts/build_content_ops_deflection_report.py \
  docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/source.csv \
  --source-format csv \
  --output docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/report.md \
  --summary-output docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/summary.json \
  --result-output docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/result.json \
  --require-output-checks \
  --json
```

The fixture uses producer-recognized fields (`ticket_id`, `created_at`,
`subject`, `message`, `pain_category`, `resolution_text`, `status`, `tags`),
so it exercises the same normalization path as hosted upload and deflection
submit. Repeated rows with the same `resolution_text` form scoped
`resolution_evidence` FAQ items; repeated rows without `resolution_text` remain
`draft_needs_review`. The proof tests assert both directions from the committed
source and generated artifacts.

## Intentional

- This is deterministic validation, not a live LLM/Gate A run. The FAQ
  deflection report is generated from source evidence and does not need a model
  route; using one would add cost without proving this lane.
- The fixture is sanitized but shaped like real provider exports: ticket IDs,
  timestamps, statuses, tags, customer wording, and agent resolution text.
- The proof keeps both lanes in one export. A resolved-only fixture would prove
  the happy path but miss the regression where question-only data gets sold as
  publishable answers.
- Archiving #1428's merged plan is included only because AGENTS.md §1g requires
  same-session teardown; it is not part of the product proof.

## Deferred

- Hosted upload/payment/delivery live run with an operator-supplied production
  export remains separate; this PR proves the deterministic report artifact,
  not Stripe/PDF/email delivery.
- In-lane residual: finish HTML normalization so raw tags cannot reach clustered
  text/output on HTML-heavy exports.
- In-lane residual: improve deterministic synonym grouping for themes with no
  shared tokens.

Parked hardening: none.

## Verification

- `python scripts/build_content_ops_deflection_report.py docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/source.csv --source-format csv --output docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/report.md --summary-output docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/summary.json --result-output docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/result.json --require-output-checks --json`
  - Result: passed; summary reports `drafted_answer_count=2`,
    `no_proven_answer_count=2`, `support_ticket_resolution_evidence_present=true`.
- `python -m pytest tests/test_content_ops_deflection_resolution_live_proof.py -q`
  - Result: `3 passed in 0.08s`.
- `python -m json.tool docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/summary.json >/dev/null && python -m json.tool docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/result.json >/dev/null`
  - Result: passed.
- `bash scripts/run_extracted_pipeline_checks.sh`
  - Result: `3551 passed, 10 skipped, 1 warning in 52.49s`; wrapper completed.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/extraction/validation/deflection_resolution_evidence_live_proof_2026-06-09.md` | 69 |
| `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/report.md` | 119 |
| `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/result.json` | 117 |
| `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/source.csv` | 13 |
| `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/summary.json` | 20 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Deflection-Resolution-Evidence-Live-Proof.md` | 145 |
| `plans/archive/PR-Deflection-Resolution-Evidence-Absent-Assertions.md` | 0 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_content_ops_deflection_resolution_live_proof.py` | 158 |
| **Total** | **645** |

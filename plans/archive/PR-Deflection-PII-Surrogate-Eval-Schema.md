# PR-Deflection-PII-Surrogate-Eval-Schema

## Why this slice exists

#1742 locks the next privacy-testing arc as corpus-first: the scrub mechanism is
implemented, but there is no derived, normalized, surrogated eval artifact that
can later feed recall/precision scoring. Without that artifact contract, the
next harness would either test synthetic-from-nothing strings again or require
committing raw customer tickets, both of which #1742 explicitly rejects.

Root cause: the measurement lane has no safe persisted data boundary. Raw
source labeling is necessarily operator-gated and touches real PII, while the
future scorer needs a versionable corpus that contains no real PII but still
preserves ticket shape and span labels. This PR fixes the upstream corpus
contract and deterministic surrogate artifact builder; it does not build the
full recall/precision scorer.

Review update root cause: the first builder version trusted complete upstream
labeling and only replaced labeled spans. That left the versionable artifact's
surrogate-only guarantee dependent on a perfect human label set. This update
fixes the root at the artifact builder by re-scanning rendered output, rejecting
raw label-span residue, avoiding raw/surrogate equality, and failing closed
before an artifact is emitted.

Diff budget note: this slice may exceed the 400 LOC soft cap because the schema,
surrogation builder, CLI, committed tiny fixture, negative-path tests, manifest,
and CI enrollment are one safety unit. Splitting the validator from the fixture
would leave the "no raw PII persisted" boundary unproven.

## Scope (this PR)

Ownership lane: deflection/pii-recall-precision-testing
Slice phase: Functional validation

1. Add a pure extracted-content module for the `deflection_pii_eval_corpus.v1`
   surrogate artifact schema.
2. Add deterministic surrogation from already-labeled local records into a
   normalized, versionable eval artifact whose labels point at surrogate spans,
   not raw PII spans.
3. Add an operator-run CLI that reads labeled local JSON and writes the
   surrogate-only artifact; raw source remains transient and caller-owned.
4. Commit one tiny synthetic surrogate fixture that exercises high, medium, low,
   cue-prefixed name, cue-less name, private-note, and must-survive tokens.
5. Add tests for happy path, malformed/missing decoded input, no-raw-span output,
   must-survive preservation, repeated spans, and CLI failure on invalid labels.
6. Add a fail-closed output leak guard for high-confidence PII-shaped residue,
   reject empty corpora, avoid raw/surrogate equality, and map must-survive
   offsets from raw positions through replacements.

### Review Contract

- Acceptance criteria:
  - [ ] Artifact schema is explicit: tickets, normalized fields, labels,
        severities, name subtype, origin field, and must-survive tokens.
  - [ ] Output labels use surrogate spans and updated start/end offsets; raw
        input PII spans are not persisted in artifact text, labels, errors, or
        metadata.
  - [ ] High/medium/low severity defaults follow #1742's class table.
  - [ ] `person_name` labels preserve `cue_prefixed` vs `cue_less`.
  - [ ] Must-survive tokens remain unchanged and are recorded for later
        precision scoring.
  - [ ] Rendered output is re-scanned before artifact emission; unlabeled
        PII-shaped residue or raw labeled spans outside recorded surrogate
        positions fail closed with sanitized errors.
  - [ ] Empty record sets are rejected rather than emitted as valid zero-case
        eval corpora.
  - [ ] Decoded malformed input is rejected as structured errors rather than
        crashing on `None`, non-object rows, non-string fields, or missing
        labels.
  - [ ] No scorer/harness, report renderer, snapshot projection, or NER logic
        is added in this PR.
- Affected surfaces: one extracted package module, one CLI, one tiny surrogate
  fixture, focused tests, manifest/runner enrollment, and this plan.
- Risk areas: accidentally persisting raw PII labels, losing span alignment
  during replacement, or pretending this replaces the operator-labeled real
  corpus.
- Reviewer rules triggered: R1, R2, R6, R9, R10, R12, R14.

### Files touched

- `.github/workflows/extracted_pipeline_checks.yml`
- `docs/extraction/validation/fixtures/deflection_pii_surrogate_eval_tiny.json`
- `extracted_content_pipeline/deflection_pii_eval_corpus.py`
- `extracted_content_pipeline/manifest.json`
- `plans/PR-Deflection-PII-Surrogate-Eval-Schema.md`
- `scripts/build_deflection_pii_surrogate_eval_corpus.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py`

## Mechanism

`extracted_content_pipeline/deflection_pii_eval_corpus.py` defines a pure,
deterministic builder:

- input: already-labeled local records with normalized text fields, PII label
  spans, and must-survive tokens;
- validation: tolerant decoded-input checks return structured errors for
  missing or malformed labels/fields instead of throwing;
- surrogation: each label span is replaced with a class-matched synthetic value
  from a deterministic pool, then labels are rewritten to the surrogate span and
  new offsets;
- safety guard: rendered fields are re-scanned before emission. Any raw label
  span residue or high-confidence PII-shaped text outside recorded surrogate
  label positions returns sanitized errors and no artifact;
- must-survive mapping: precision tokens are resolved against raw text by
  occurrence, then mapped through replacement offsets so repeated tokens do not
  collapse to the first occurrence;
- output: `deflection_pii_eval_corpus.v1` JSON with only normalized ticket
  fields, surrogate labels, must-survive records, and summary counts.

`scripts/build_deflection_pii_surrogate_eval_corpus.py` is a thin CLI wrapper
around the pure builder. It exits non-zero if validation errors exist and never
prints input text. The tiny committed fixture under
`docs/extraction/validation/fixtures/` is generated from synthetic labeled
records and proves the persisted artifact shape without using raw customer
source.

## Intentional

- No raw-source labeling UI or auto-labeler. #1742 says the real raw PII step is
  operator-gated; this PR starts after labels exist.
- No recall/precision scorer yet. A scorer without the corpus contract would be
  another harness around a missing input.
- No model/NER dependency. Cue-less names are represented and labeled so the
  future harness can expose that ceiling; closing the ceiling is a later slice.
- The output leak guard is deliberately deterministic and high-confidence:
  email, phone, SSN, payment-card, title-case street-address, and cue-prefixed
  person-name shapes. Open-set cue-less name discovery remains a measured
  ceiling for the future harness.
- No real customer export or legacy `claude/pr-deflection-zendesk-product-proof-corpus`
  branch reuse. This slice creates the new surrogate artifact path, not the old
  proof-corpus lane.

## Deferred

- Operator supplies/chooses the real source and labeling workflow for the
  derived corpus.
- Larger versioned eval artifact built from operator-labeled, surrogated real
  source.
- Phase 2 scoring harness: run the full deflection pipeline and score recall /
  precision per PII class x output surface.
- Advisory CI artifact and later headline KPI gate once corpus + harness are
  stable.

Parked hardening: none.

## Verification

- Command: `python -m py_compile` on
  `extracted_content_pipeline/deflection_pii_eval_corpus.py`,
  `scripts/build_deflection_pii_surrogate_eval_corpus.py`, and
  `tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py` -- passed.
- Command: `python -m pytest` on
  `tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py` with `-q` --
  11 passed.
- Command: `scripts/audit_extracted_pipeline_ci_enrollment.py` through python --
  passed; 188 matching tests enrolled.
- Command: `scripts/run_extracted_pipeline_checks.sh` through bash -- passed;
  reasoning core 295 passed, extracted content 4738 passed / 15 skipped.
- Command: `scripts/push_pr.sh` with
  `tmp/pr_body_deflection_pii_surrogate_eval_schema.md` and branch push args --
  passed; local PR review passed in the managed pre-push hook.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/extracted_pipeline_checks.yml` | 2 |
| `docs/extraction/validation/fixtures/deflection_pii_surrogate_eval_tiny.json` | 140 |
| `extracted_content_pipeline/deflection_pii_eval_corpus.py` | 580 |
| `extracted_content_pipeline/manifest.json` | 3 |
| `plans/PR-Deflection-PII-Surrogate-Eval-Schema.md` | 176 |
| `scripts/build_deflection_pii_surrogate_eval_corpus.py` | 68 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_content_ops_deflection_pii_surrogate_eval_corpus.py` | 432 |
| **Total** | **1402** |

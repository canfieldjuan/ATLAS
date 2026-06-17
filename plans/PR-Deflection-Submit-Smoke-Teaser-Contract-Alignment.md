# PR-Deflection-Submit-Smoke-Teaser-Contract-Alignment

## Why this slice exists

Issue #1612 is now green on the ATLAS-side paid artifact/report-model/PDF/export
proof path, but the submit handoff smoke still has one parked false-positive
from the live-artifact proof: it reports `teaser.full_answer.answer` and
`teaser.full_answer.steps` as forbidden paid-report leaks even though
`docs/frontend/content_ops_faq_report_contract.md` explicitly permits answer
text and steps inside `snapshot.teaser.full_answer`.

Root cause: the smoke uses a path-insensitive forbidden-key denylist for the
snapshot payload. That is correct for `answer` and `steps` almost everywhere,
but it cannot represent the one contracted exception at
`$.teaser.full_answer`. This slice fixes the root by making the smoke's
forbidden-field detector path-aware. It does not loosen the paywall boundary
globally.

## Scope (this PR)

Ownership lane: content-ops/deflection-full-report-qa
Slice phase: Production hardening

1. Update `scripts/smoke_content_ops_deflection_submit_handoff.py` so
   `answer` and `steps` are allowed only at `$.teaser.full_answer.answer` and
   `$.teaser.full_answer.steps`, and only when the full-answer object carries
   scoped resolution evidence markers.
2. Keep the existing denylist active for the same keys everywhere else,
   including `top_questions`, `locked_questions`, teaser previews, and any
   unexpected nested object.
3. Add focused boundary tests covering:
   - contracted `teaser.full_answer.answer` / `steps` passes;
   - draft or unscoped `teaser.full_answer.answer` / `steps` fails;
   - dotted-key and nested-alias paths fail;
   - `top_questions[0].answer` still fails;
   - `teaser.previews[0].answer` and `teaser.previews[0].steps` still fail.
4. Remove the drained HARDENING.md item for the submit-smoke teaser false
   positive.

### Files touched

- `HARDENING.md`
- `plans/PR-Deflection-Submit-Smoke-Teaser-Contract-Alignment.md`
- `scripts/smoke_content_ops_deflection_submit_handoff.py`
- `tests/test_smoke_content_ops_deflection_submit_handoff.py`

### Review Contract

- Acceptance criteria:
  - [ ] Submit smoke accepts `answer` and `steps` at the contract-approved
        `$.teaser.full_answer` path only for scoped resolution-backed teaser
        answers.
  - [ ] Submit smoke still rejects paid-answer fields outside that exact path.
  - [ ] The parked HARDENING.md item is removed only because this PR drains it.
- Affected surfaces: operator submit-smoke validation and #1612 proof
  interpretation.
- Risk areas: paywall contract precision, false positives, false negatives.
- Reviewer rules triggered: R1, R2, R10, R14

## Mechanism

Keep the existing `FORBIDDEN_SNAPSHOT_KEYS` denylist, but add an exact structural
path allowlist for the contracted teaser body fields. `_forbidden_key_paths`
walks the snapshot recursively with path segments, so a literal key such as
`teaser.full_answer` cannot alias the real nested path. The exception also
requires the parent full-answer object to carry `answer_evidence_status:
resolution_evidence` and `resolution_evidence_scope: scoped` before body fields
are exempted.

This makes the detector precise without teaching it the whole snapshot schema:
the smoke remains a leak guard, not a full TypeScript contract validator.

## Intentional

- No change to the producer contract. The docs already permit
  `teaser.full_answer.answer` and `teaser.full_answer.steps`.
- No blanket exception for key names under `teaser`. Preview entries still
  withhold body text, so `teaser.previews[*].answer` and `steps` remain leaks.
- No buyer hosted-result proof in this ATLAS slice; that remains the
  atlas-portfolio/web lane unless reassigned.

## Deferred

- atlas-portfolio/web buyer hosted-result proof remains outside this ATLAS
  session unless explicitly reassigned.

Parked hardening: drains `Align deflection submit smoke forbidden snapshot paths
with teaser contract` from `HARDENING.md`.

## Verification

- Focused submit-smoke pytest for `tests/test_smoke_content_ops_deflection_submit_handoff.py`.
  - 47 passed.
- Python compile check for `scripts/smoke_content_ops_deflection_submit_handoff.py`
  and `tests/test_smoke_content_ops_deflection_submit_handoff.py`.
  - passed.
- Full extracted pipeline bundle via `scripts/run_extracted_pipeline_checks.sh`.
  - extracted reasoning core: 295 passed.
  - extracted content pipeline: 4636 passed, 10 skipped, 1 existing torch warning.
- Pending before PR open:
  - push-wrapper local PR review with `tmp/pr_body_deflection_submit_smoke_teaser_contract_alignment.md`.

## Estimated diff size

| File | LOC |
|---|---:|
| `HARDENING.md` | 9 |
| `plans/PR-Deflection-Submit-Smoke-Teaser-Contract-Alignment.md` | 111 |
| `scripts/smoke_content_ops_deflection_submit_handoff.py` | 37 |
| `tests/test_smoke_content_ops_deflection_submit_handoff.py` | 139 |
| **Total** | **296** |

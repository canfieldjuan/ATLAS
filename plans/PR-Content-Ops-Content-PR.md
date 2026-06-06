# PR — Content-Ops Content-PR Review Contract (operating-model slice 4)

## Why this slice exists

`docs/content_ops_operating_model.md` makes the **Content-PR + coverage matrix** the
anti-drift core of the review contract: a reviewer is handed a structured Content-PR
(frozen rule-pack versions + claims map + a required coverage matrix), and the verdict is
*computed* under one rule -- **no required rule passes silently**. This is the slice where
the earlier slices compose: slice 1's `ReviewDecision` is the verdict, slice 3's
`blocking_claims` feeds it, and slice 4 adds the coverage matrix + comment discipline.

It is the doc's "epic"; per the operator it ships as a single slice. It stays deterministic
and additive (no LLM, no wiring into live flows) -- the adversarial pass, calibration
library, and LLM-driven coverage generation are later slices. Diff runs over the 400-LOC
soft cap (see *Estimated diff size*); the excess is test surface for the verdict matrix,
and the integration value justifies landing it whole rather than splitting.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Vertical slice

New owned module `extracted_content_pipeline/content_pr.py` + tests. Pure value types +
pure functions; no I/O, no Atlas imports, no DB, no LLM:

- `CoverageStatus` (StrEnum) — PASS / FAIL / NOT_APPLICABLE / UNRESOLVED.
- `CommentCategory` (StrEnum) — brief / brand_rule / claim_registry / compliance /
  channel_constraint / performance_hypothesis / editorial_judgment / nit (the NIT escape
  hatch: categorized, never blocking).
- `RulePacketVersions` (frozen) — the five pinned rule-pack version stamps; `missing` /
  `is_pinned`.
- `CoverageRow` (frozen) — a required-rule row; `is_resolved` (PASS/FAIL need cited
  evidence, NOT_APPLICABLE needs none, UNRESOLVED never resolves).
- `ReviewComment` (frozen) — categorized + evidenced; a NIT may not be blocking
  (`__post_init__` guard).
- `ContentPR` (frozen) — rule packet + slice-3 claims map + coverage matrix + comments.
- `review_verdict(pr)` — computes `ReviewDecision` (auto-derives only BLOCKED /
  REVISION_REQUIRED / APPROVED); helpers `unresolved_required_rows`,
  `failing_required_rows`, `blocking_comments`, and `verdict_reasons` (the transparency
  trail).

Reviewer rules triggered: R1 (coverage rows record Pass/Fail/N-A requirement matches; the verdict enforces them), R10 (the verdict and its predicates are small, pure, and maintainable).

### Files touched

- `extracted_content_pipeline/content_pr.py`
- `extracted_content_pipeline/manifest.json`
- `plans/PR-Content-Ops-Content-PR.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_extracted_content_pr.py`

## Mechanism

`review_verdict` gates in order: an unpinned rule packet or any unresolved required
coverage row -> `BLOCKED` (the review itself is incomplete / untrustworthy; this is the
"no silent pass" rule); else any failed required row, slice-3 `blocking_claims`, or
blocking comment -> `REVISION_REQUIRED`; else `APPROVED`. The human-only states
(`APPROVED_WITH_EXCEPTION`, `ESCALATED`) are never auto-produced. Same conventions as
slices 1-3 (`StrEnum` 3.10 fallback, frozen dataclasses, `None`/non-`str` tolerated).

## Intentional

- Additive only — nothing wires this into a live review flow yet; that's a later slice.
- The verdict is **completeness + hard-failure** logic, not quality judgment: it decides
  whether the review is *complete and clean*, not whether the copy is *good* (human/market).
- BLOCKED (incomplete review) deliberately takes precedence over REVISION_REQUIRED
  (content failure): you can't demand a specific revision from an unfinished review.

## Deferred

- The adversarial second pass and the review calibration library (slice 5).
- LLM-driven coverage-row / comment generation; wiring `review_verdict` into the
  generated-assets API. Brief-snapshot persistence schema. Multi-model disagreement
  orchestration stays parked.

## Verification

- pytest `tests/test_extracted_content_pr.py`
- `scripts/check_ascii_python.sh` (run via bash) -- ASCII gate
- `scripts/check_extracted_imports.py` (run via python3) -- import structure
- `scripts/audit_extracted_pipeline_ci_enrollment.py` -- new test enrolled
- `scripts/audit_extracted_standalone.py` (--fail-on-debt) -- no Atlas runtime imports
- `scripts/audit_pr_session_drift.py` + `scripts/sync_pr_plan.py` -- plan shape/drift

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/content_pr.py` | 216 |
| `extracted_content_pipeline/manifest.json` | 3 |
| `plans/PR-Content-Ops-Content-PR.md` | 94 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_extracted_content_pr.py` | 259 |
| **Total** | **573** |

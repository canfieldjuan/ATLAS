# PR — Content-Ops Review Vocabulary (operating-model slice 1)

Ownership lane: content-ops/review-contract
Slice phase: Content operating model - slice 1 (review vocabulary)

## Why this slice exists

`docs/content_ops_operating_model.md` defines a content-review subsystem and a
"Building this: staged, not a monolith" sequence. This is **slice 1 — free reframes +
cheap enums**: the typed vocabulary every later slice consumes, landed before any
behavior changes so the expensive slices (claims map, Content-PR coverage matrix) plug
into shared names instead of inventing their own.

It is deliberately enum/value-scale and changes no existing behavior. The existing
generated-asset review API keeps its host-extensible string statuses untouched; wiring
this vocabulary into validation/routing is slice 2.

Diff total is marginally over the 400-LOC soft cap (~412). The overage is scaffolding,
not scope: the module is ~190 LOC (largely docstrings), and the rest is the plan doc's
own machine-generated tables plus thorough unit tests. The shippable surface stays tiny.

## Scope (this PR)

New owned module `extracted_content_pipeline/review_contract.py` (flat-module
convention, per `brand_voice.py`) plus unit tests. It provides:

- `RiskTier` (StrEnum: LOW/MEDIUM/HIGH/CRITICAL) — the brief-set editorial review-burden
  tier. Distinct from `extracted_quality_gate.RiskLevel` (a computed safety score);
  same labels, different domain. Docstring cross-references both.
- `ReviewDecision` (StrEnum) — BLOCKED / REVISION_REQUIRED / APPROVED /
  APPROVED_WITH_EXCEPTION / ESCALATED.
- `FailureCategory` (StrEnum) — the verdict failure taxonomy (15 categories).
- `GateStage` (StrEnum) — the four-part gate split: SCHEMA (3A) /
  CLAIMS_COMPLIANCE (3B) / MODEL_ASSISTED (3C) / HUMAN_EDITOR (3D).
- `REQUIRED_REVIEW_BY_TIER` — frozen mapping tier -> ordered gate stages (the doc's
  risk-tier table as pure data) + `required_stages_for(tier)`.
- `ExceptionRecord` (frozen dataclass) — rule/reason/owner/expiration/should_update_rule
  + `is_active(as_of)`.
- `recurring_failure_categories(categories, *, threshold=3)` — the flywheel
  "3+ same-reason misses -> candidate for a new gate" helper.


### Files touched

- `extracted_content_pipeline/manifest.json`
- `extracted_content_pipeline/review_contract.py`
- `plans/PR-Content-Ops-Review-Vocabulary.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_extracted_content_review_contract.py`

## Mechanism

Pure value types and pure functions. `StrEnum` with the same Python-3.10 import fallback
used in `extracted_quality_gate/types.py`. No I/O, no Atlas imports, no DB, no LLM. Each
type ships one small tested behavior so nothing is inert: `required_stages_for`,
`ExceptionRecord.is_active`, `recurring_failure_categories`. `__init__.py` is empty by
convention (full-path imports), so no export edit.

## Intentional

- Additive only — zero change to existing review status strings or the
  generated-assets API. Verified by not touching those files.
- `RiskTier` kept separate from `quality_gate.RiskLevel` rather than reused, because one
  is an editorial routing tier and the other a computed safety score; conflating them
  would couple review burden to the safety sensor.
- `REQUIRED_REVIEW_BY_TIER` is pure data (the table); the *enforcement/routing* that
  consumes it is slice 2.

## Deferred

- Wiring `ReviewDecision` into `api/generated_assets.py` status validation (slice 2).
- Risk-tier routing / experiment-contract + triage schemas (slice 2).
- Claims map (slice 3); Content-PR coverage matrix + comment/evidence types (slice 4);
  calibration library + adversarial pass (slice 5).
- Multi-model disagreement orchestration (parked, out of scope per the doc).

## Verification

- pytest `tests/test_extracted_content_review_contract.py` (17 tests)
- `scripts/check_ascii_python.sh` (run via bash) -- ASCII gate
- `scripts/check_extracted_imports.py` (run via python3) -- import structure
- `extracted/_shared/scripts/forbid_atlas_reasoning_imports.py` + `scripts/audit_extracted_standalone.py` (--fail-on-debt) -- both clean
- import-sanity: `python3 -c "import extracted_content_pipeline.review_contract as m"`

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/manifest.json` | 3 |
| `extracted_content_pipeline/review_contract.py` | 190 |
| `plans/PR-Content-Ops-Review-Vocabulary.md` | 94 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_extracted_content_review_contract.py` | 131 |
| **Total** | **419** |

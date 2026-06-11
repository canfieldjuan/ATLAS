# PR-Content-Ops-Calibration-Library

## Why this slice exists

Operating-model tracker #1338 lists slice 5 (**calibration library + adversarial
pass**) as the last unbuilt deterministic core, and #1353/#1435 keep the
marketer-verify MCP surface deliberately thin and verify-only until the
reviewer-anchoring pieces exist. `docs/content_ops_operating_model.md` names the
**review calibration library** as "the missing anti-drift piece": without curated
worked examples, "brand voice" is a seance, and both human and model reviewers
have nothing to anchor on. This slice lands the deterministic half of slice 5 --
the calibration set as a pure typed store -- and defers the adversarial pass to
5b. It closes the operator's scope nod on #1338 ("land the deterministic core and
defer the LLM, like slice 3").

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Vertical slice

1. Add `extracted_content_pipeline/calibration_library.py`: the `CalibrationLabel`
   vocabulary, a frozen `CalibrationExample`, an immutable `CalibrationLibrary`
   query container, and a converter that turns a slice-1 override into an example.
2. Register the new module as package-owned in the manifest.
3. Enroll the new test in the extracted-pipeline CI runner.

### Files touched

- `extracted_content_pipeline/calibration_library.py`
- `extracted_content_pipeline/manifest.json`
- `plans/PR-Content-Ops-Calibration-Library.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_extracted_content_calibration_library.py`

### Review Contract

Acceptance criteria:
- The module is pure: no I/O, no Atlas imports, no DB, no LLM (only in-package
  imports from `review_contract`).
- Label polarity is data-driven and value-based: a decoded string label
  classifies the same as the enum member.
- Queries return new tuples in input order and never mutate the set.
- `by_failure_category` / `by_verdict` return empty for a `None` argument and
  never sweep in uncategorized examples.
- An override (`ExceptionRecord`) converts to a `BORDERLINE`,
  `APPROVED_WITH_EXCEPTION`, `source="override"` example carrying the override
  reason, and is then queryable like any curated example.
- Records tolerate decoded input (`None`/non-str excerpt/reasoning count as
  missing, never raise).

Affected surfaces: extracted_content_pipeline package only; no host wiring, no
MCP tool, no route, no DB. The marketer-verify MCP surface is unchanged.

Risk areas: value-vs-identity enum comparison (the recurring slice-1..4 trap);
None-argument query branches.

Reviewer rules triggered: R1, R2, R10, R14.

## Mechanism

`CalibrationLabel` is a `StrEnum` of the doc's example labels. Two module-level
frozensets (`_POSITIVE_LABELS`, `_NEGATIVE_LABELS`) encode polarity as data;
`BORDERLINE` is in neither (the studied judgment call). The `is_positive_label` /
`is_negative_label` helpers test set membership, which is value-based for
`StrEnum`, so a plain decoded string classifies identically to the enum member --
the same robustness contract the sibling slices use.

`CalibrationExample` is a frozen record (id, excerpt, label, reasoning, optional
verdict + failure_category, provenance source). Its predicate for whether an
anchor actually teaches requires both excerpt and reasoning to be non-whitespace
text; `None` counts as missing rather than raising.

`CalibrationLibrary` is a frozen tuple-backed container. Its query helpers filter
by label, failure category, verdict, and polarity, report the distinct labels
present, and report required-but-absent labels (the set's blind spots) in order
and de-duplicated. The `None`-argument branches on the category/verdict queries
return empty so an uncategorized example is never silently grouped.

`example_from_exception` is the compounding seam: it maps a slice-1
`ExceptionRecord` (an approve-with-exception override) into a calibration example
-- defaulting to a `BORDERLINE` anchor, fixing the verdict to
`APPROVED_WITH_EXCEPTION`, carrying the override reason as the teaching
reasoning, and stamping provenance `override` -- so the judgment layer compounds
exactly as the doc's section-5 flywheel prescribes.

## Intentional

- **No host/MCP wiring.** This is the deterministic store; scoring a fresh draft
  *against* the anchors (the LLM step) and exposing it through the verifier are
  deferred. Matches the slice-3 precedent (land the data, defer the extractor).
- **Polarity is a fixed data table, not per-example.** A label's meaning is
  global; encoding it once keeps curation from drifting example by example.
  `BORDERLINE` intentionally has no polarity.
- **The library is immutable; curation builds a new one.** Consistent with the
  frozen-record convention across slices 1/3/4; avoids a mutable-set aliasing
  surface in a value module.
- **`example_from_exception` defaults the label rather than inferring it.** An
  override is a judgment call, so `BORDERLINE` is the safe default; the caller
  can pass `OVERCLAIM`/etc. when the override has a known failure shape.

## Deferred

- **Slice 5b -- adversarial pass:** the data model for a second independent
  review pass + a deterministic disagreement/merge helper between two passes
  (operating-model section 4). Named in #1338; unlocked by this slice landing.
- **LLM scoring against anchors:** selecting the nearest anchors for a fresh
  draft and the model-assisted drift judgment (operating-model stage 3C) -- the
  step that actually consumes this set. Gated behind the #1435 reliability work.
- **Host persistence / a curated seed set:** this slice ships the type + queries,
  not a populated library or a DB table.
- Parked hardening: none.

## Verification

- Reviewer rules triggered: R1, R2, R10, R14.
- Passed: python3 -m py_compile of the new module and test -- OK.
- Passed: pytest of the new slice-5a test file -- 24 passed.
- Passed: pytest of slice 5a plus sibling slices 1, 3, 4 -- 72 passed, no
  sibling-slice regression.
- Passed: python3 scripts/audit_extracted_pipeline_ci_enrollment.py -- OK, 166
  matching tests are enrolled (includes the new test).
- Passed: python3 scripts/audit_extracted_standalone.py --fail-on-debt -- Atlas
  runtime import findings: 0.
- Passed: bash scripts/check_ascii_python.sh -- ASCII check passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/calibration_library.py` | 241 |
| `extracted_content_pipeline/manifest.json` | 3 |
| `plans/PR-Content-Ops-Calibration-Library.md` | 135 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_extracted_content_calibration_library.py` | 248 |
| **Total** | **628** |

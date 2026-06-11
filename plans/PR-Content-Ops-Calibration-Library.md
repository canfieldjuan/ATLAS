# PR-Content-Ops-Calibration-Library

## Why this slice exists

Operating-model tracker #1338 lists slice 5 (**calibration library + adversarial
pass**) as the last unbuilt deterministic core, and #1353/#1435 keep the
marketer-verify MCP surface deliberately thin and verify-only until the
reviewer-anchoring pieces exist. `docs/content_ops_operating_model.md` names the
**review calibration library** as "the missing anti-drift piece": without curated
worked examples, "brand voice" is a seance, and both human and model reviewers
have nothing to anchor on. This slice lands the full deterministic core of
slice 5 exactly as the tracker row defines it -- the calibration set as a pure
typed store (5a) plus the adversarial-pass data model and deterministic
disagreement/merge helpers (5b) -- and defers the LLM steps (anchor scoring,
finding-producing prompts, disagreement orchestration). It closes the
operator's scope nod on #1338 ("land the deterministic core and defer the LLM,
like slice 3"). The diff is over the 400-LOC soft cap because the two halves
are one tracker row and ~55% of the diff is the detection-coverage test suite
the AGENTS 3i contract requires; the production modules total ~400 LOC.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Vertical slice

1. Add `extracted_content_pipeline/calibration_library.py`: the `CalibrationLabel`
   vocabulary, a frozen `CalibrationExample`, an immutable `CalibrationLibrary`
   query container, and a converter that turns a slice-1 override into an example.
2. Add `extracted_content_pipeline/adversarial_pass.py`: the
   `AdversarialFindingCategory` vocabulary (the doc's target list), frozen
   `AdversarialFinding` / `AdversarialPass` records, deterministic
   corroboration/disagreement/merge helpers between two passes, and a converter
   that turns a finding into a never-blocking slice-4 review comment.
3. Register both new modules as package-owned in the manifest.
4. Enroll both new tests in the extracted-pipeline CI runner.

### Files touched

- `extracted_content_pipeline/adversarial_pass.py`
- `extracted_content_pipeline/calibration_library.py`
- `extracted_content_pipeline/manifest.json`
- `plans/PR-Content-Ops-Calibration-Library.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_extracted_content_adversarial_pass.py`
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
- An adversarial finding converts to a review comment that is **never
  blocking** (the "still not a judge" invariant), with the finding category
  carried in the message prefix.
- Corroboration is the category intersection of two passes; disagreement is the
  symmetric difference; merge de-duplicates only exact duplicates and preserves
  first-then-second order.

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

The adversarial-pass module (5b) models the doc's second independent pass.
`AdversarialFindingCategory` is the doc's target list (overclaim, ambiguity,
reader objection, promise/CTA mismatch, generic stretch, missing proof, voice
slip). `AdversarialFinding` is substantiated only when it carries both an
objection and quoted evidence -- the filter that keeps a chatty pass from
flooding review. `AdversarialPass` records the prompt/model identity so two
passes are distinguishable. The deterministic helpers compute the corroborated
categories (set intersection -- the strongest signal), the disagreement surface
(symmetric difference -- the *data* for the parked orchestration), and an
exact-duplicate-only merge. `comment_from_finding` is where "still not a judge"
is enforced: every finding becomes a `blocking=False` review comment (voice
slips route to the brand-rule lane, everything else to editorial judgment), so
only the accountable editor can escalate a model objection to blocking.

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
- **`comment_from_finding` cannot produce a blocking comment.** The doc is
  explicit that the adversarial pass is not a judge; hard-coding
  `blocking=False` at the seam makes the invariant unbypassable rather than
  conventional.
- **Merge collapses only exact duplicates.** Two differently-worded objections
  in the same category are both kept; deciding they are "the same" is a
  judgment call this module refuses to make.
- **Disagreement is computed, not acted on.** Routing an A-pass/B-fail split to
  a human and logging the override is the parked orchestration (operating-model
  section 4); this slice ships only the deterministic data it would consume.

## Deferred

- **LLM scoring against anchors:** selecting the nearest anchors for a fresh
  draft and the model-assisted drift judgment (operating-model stage 3C) -- the
  step that actually consumes the calibration set. Gated behind the #1435
  reliability work.
- **Finding-producing prompts:** the adversarial prompts/models that emit
  `AdversarialFinding` rows are the LLM step, same deferral as above.
- **Model-disagreement orchestration:** parked per the doc until slices 1-5
  prove out.
- **Host persistence / a curated seed set:** this slice ships the types +
  queries, not a populated library or a DB table.
- Parked hardening: none.

## Verification

- Reviewer rules triggered: R1, R2, R10, R14.
- Passed: python3 -m py_compile of both new modules and tests -- OK.
- Passed: pytest of the slice-5a test file -- 24 passed; slice-5b test file --
  17 passed.
- Passed: pytest of slice 5 (both halves) plus sibling slices 1, 3, 4 -- 89
  passed, no sibling-slice regression.
- Passed: python3 scripts/audit_extracted_pipeline_ci_enrollment.py -- OK, 167
  matching tests are enrolled (includes both new tests).
- Passed: python3 scripts/audit_extracted_standalone.py --fail-on-debt -- Atlas
  runtime import findings: 0.
- Passed: bash scripts/check_ascii_python.sh -- ASCII check passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/adversarial_pass.py` | 181 |
| `extracted_content_pipeline/calibration_library.py` | 241 |
| `extracted_content_pipeline/manifest.json` | 6 |
| `plans/PR-Content-Ops-Calibration-Library.md` | 178 |
| `scripts/run_extracted_pipeline_checks.sh` | 2 |
| `tests/test_extracted_content_adversarial_pass.py` | 223 |
| `tests/test_extracted_content_calibration_library.py` | 248 |
| **Total** | **1079** |

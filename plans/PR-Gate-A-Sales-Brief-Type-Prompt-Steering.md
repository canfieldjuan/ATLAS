# PR-Gate-A-Sales-Brief-Type-Prompt-Steering

## Why this slice exists

PR #1364 fixed the Gate A sales-brief label drift: a requested
`brief_type=renewal` now persists as `renewal` even when the model returns
`pre_call`. Its review left a non-blocking NIT that becomes the next hardening
slice: the label is locked, but the prompt still does not tell the model which
sales motion to write for. A renewal brief can be correctly labeled `renewal`
while reading like a generic pre-call brief.

This slice closes that content-quality gap by threading the explicit
per-call requested `brief_type` into the sales-brief prompt so generated
content is steered toward the same sales motion that persistence records.

## Scope (this PR)

Ownership lane: content-ops/gate-a-output-quality
Slice phase: Production hardening

1. Thread a non-empty per-call `default_brief_type` into the sales-brief LLM
   user prompt.
2. Add brief-type-specific guidance for the known motions:
   `pre_call`, `renewal`, `displacement`, and `discovery`.
3. Add a grounding guard so the motion-specific guidance does not invite
   invented contract dates, renewal windows, competitor names, or timelines.
4. Keep no-request behavior unchanged: when no per-call brief type is supplied,
   the prompt stays at its existing generic/default shape.
5. Add focused tests proving renewal prompt steering and no-request backward
   compatibility.

### Review Contract

- Acceptance criteria:
  - [ ] A generation call with `default_brief_type="renewal"` includes an
        explicit requested brief-type block in the user prompt.
  - [ ] The renewal prompt guidance mentions renewal-stage concerns such as
        retention, expansion, contract timing, or renewal risk.
  - [ ] The brief-type guidance tells the model to use only supplied
        opportunity evidence and not invent contract/timeline details.
  - [ ] A generation call without a per-call `default_brief_type` keeps the
        existing prompt free of a requested brief-type block.
  - [ ] Variant-angle prompt steering still works alongside brief-type
        steering.
  - [ ] Persistence precedence from #1364 remains unchanged.
- Affected surfaces: extracted content pipeline sales-brief prompt construction
  and focused sales-brief generation tests.
- Risk areas: prompt behavior/backcompat, accidentally changing persisted
  `brief_type` precedence, over-scoping into live Gate A reruns.
- Reviewer rules triggered: R1, R2, R10.

### Files touched

- `extracted_content_pipeline/sales_brief_generation.py`
- `plans/PR-Gate-A-Sales-Brief-Type-Prompt-Steering.md`
- `tests/test_extracted_sales_brief_generation.py`

## Mechanism

`SalesBriefGenerationService.generate(...)` already receives the run-plan
`default_brief_type`. Today it passes that value only to `_build_draft(...)`.
This PR also passes the non-empty per-call value into `_generate_one(...)`,
then into `_sales_brief_user_prompt(...)`.

The user prompt gains a small block only when the caller explicitly requested a
brief type:

```text
Requested brief type:
- renewal: write for renewal-stage retention, expansion, contract timing, and
  renewal-risk discussion.
Use only the supplied opportunity evidence; do not invent contract dates,
renewal windows, competitor names, or timelines that are not in the data.
```

Unknown values still get a generic "match this requested sales motion" line so
future string labels are not silently ignored. Persistence keeps the #1364
precedence:

```text
per-call requested brief type -> model brief_type -> service config default
```

## Intentional

- This does not add enum validation. Existing sales-brief surfaces already
  accept string labels, and unknown labels can still carry useful custom sales
  motions.
- This does not change the sales-brief skill markdown. The run-plan value is
  per-call state, so the user prompt is the narrowest place to add it.
- This does not run a live Gate A generation. The change is prompt-contract
  wiring with focused unit coverage; live acceptance belongs after the remaining
  Gate A quality slices.
- This does not change persisted `brief_type` precedence from #1364.
- This adds the review-requested grounding guard in the user prompt instead of
  weakening the motion-specific guidance; the prompt can still steer toward a
  motion, but only from supplied opportunity evidence.
- Cross-layer caller hints were inspected for `SalesBriefGenerationService`.
  The changed parameters are optional keyword/defaulted prompt inputs, so
  existing constructor/generate callers keep the old behavior; the no-request
  prompt assertion and full extracted suite cover that unchanged path.

## Deferred

- Gate A brand voice enforcement: resolve the parked second-person miss.
- Gate A landing-page distinctness: make whole-page variants meaningfully
  different, not only hero-headline variants.
- Gate A blog prose quality: resolve debug-style source narration.
- Gate A messy-ticket rerun: rerun the live proof on noisy support-ticket data
  after structural and prompt-quality fixes are in place.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_extracted_sales_brief_generation.py -q` -
  31 passed in 0.11s after adding the grounding-guard assertion.
- `bash scripts/validate_extracted_content_pipeline.sh` - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline`
  - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed.
- `bash scripts/check_ascii_python.sh` - passed.
- `bash scripts/run_extracted_pipeline_checks.sh` - 3251 passed, 10 skipped,
  1 warning in 55.03s.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/gate-a-sales-brief-type-prompt-steering-pr-body.md`
  - passed; advisory cross-layer caller hints inspected and documented above.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/sales_brief_generation.py` | 44 |
| `plans/PR-Gate-A-Sales-Brief-Type-Prompt-Steering.md` | 135 |
| `tests/test_extracted_sales_brief_generation.py` | 36 |
| **Total** | **215** |

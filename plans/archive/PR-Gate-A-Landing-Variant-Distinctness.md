# PR-Gate-A-Landing-Variant-Distinctness

## Why this slice exists

Gate A live output-quality proof found that landing-page variants passed
structural and brand-voice audits while still being weak product variants: the
three pages had distinct hero headlines, but shared the same title and much of
the same body copy. `HARDENING.md` records this as "Landing-page variants pass
audits but are not meaningfully distinct."

This slice strengthens first-pass landing-page variant instructions so the
variant angle changes whole-page framing, not only the hero line. The live Gate
A rerun remains deferred; this PR closes the prompt/scaffold weakness that made
the prior run too similar.

## Scope (this PR)

Ownership lane: content-ops/gate-a-output-quality
Slice phase: Production hardening

1. Strengthen the landing-page variant prompt to require distinct whole-page
   framing across title, hero, section order/emphasis, section titles, body
   leads, `metadata.answer_summary`, FAQ/objection framing, and CTA support
   copy.
2. Keep the same evidence, CTA intent, and truthfulness limits; this is not a
   license to invent claims to make variants feel different.
3. Add focused tests proving the prompt warns against hero-only variants and
   names the page surfaces that must vary.

### Review Contract

- Acceptance criteria:
  - [ ] Variant prompts explicitly say not to stop at a hero/headline swap.
  - [ ] Variant prompts require the angle to shape title, hero, section
        order/emphasis, section titles, `metadata.answer_summary`, body leads,
        FAQ/objection framing, and CTA support copy.
  - [ ] Variant prompts preserve the existing grounding constraints around
        campaign facts, supplied evidence, CTA intent, and truthfulness limits.
  - [ ] Non-variant landing-page prompts stay compact and do not receive the
        variant-only distinctness instructions.
- Affected surfaces: landing-page generation prompt text and focused extracted
  landing-page smoke tests.
- Risk areas: prompt-only prevention, over-broad instructions that encourage
  invented claims, and conflating this Gate A slice with #1374's review/coverage
  row lane.
- Reviewer rules triggered: R1, R2, R10.

### Files touched

- `extracted_content_pipeline/landing_page_generation.py`
- `plans/PR-Gate-A-Landing-Variant-Distinctness.md`
- `tests/test_extracted_landing_page_generation_smoke.py`

## Mechanism

The existing variant prompt already threads the selected deterministic
`VariantAngle` into the user message. This slice tightens that block:

```text
Variant angle:
- <angle>
Use this angle to make the whole landing page meaningfully distinct...
Do not stop at changing only the hero headline...
```

The prompt names concrete page surfaces that must be influenced by the angle,
including `metadata.answer_summary` so repair guidance that requires body copy
to start with that summary preserves variant distinctness instead of undoing
it. The prompt repeats the evidence/CTA/truthfulness constraints. The tests
assert both sides: variant calls receive the whole-page distinctness contract,
and non-variant calls do not.

## Intentional

- This is prompt prevention, not a cross-draft similarity gate. The generator
  runs one landing page at a time and does not have the other variants'
  rendered copy in scope.
- This does not rerun the live Gate A proof. The next validation slice should
  run the configured cloud/OpenRouter live smoke and inspect generated samples.
- This does not touch #1374's quality-gate coverage rows or review-service
  lane.

## Deferred

- Gate A live rerun to prove landing-page variants are meaningfully distinct in
  generated samples.
- Gate A messy-ticket grounding rerun remains deferred until the known output
  quality misses are structurally addressed.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_extracted_landing_page_generation_smoke.py -q` - PASS (`3 passed`).
- `bash scripts/validate_extracted_content_pipeline.sh` - PASS.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - PASS.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - PASS (`Atlas runtime import findings: 0`).
- `bash scripts/check_ascii_python.sh` - PASS.
- `bash scripts/run_extracted_pipeline_checks.sh` - PASS (`extracted_reasoning_core`: `295 passed`; `extracted_content_pipeline`: `3322 passed, 10 skipped`; one `pynvml` warning).
- `bash scripts/local_pr_review.sh --current-pr-body-file tmp/gate-a-landing-variant-distinctness-pr-body.md` - PASS (no non-diff caller references).

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/landing_page_generation.py` | 6 |
| `plans/PR-Gate-A-Landing-Variant-Distinctness.md` | 106 |
| `tests/test_extracted_landing_page_generation_smoke.py` | 32 |
| **Total** | **144** |

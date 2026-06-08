# PR-Gate-A-Brand-Voice-Second-Person-Enforcement

## Why this slice exists

Gate A live output-quality proof found that the selected brand voice profile
reached the LLM prompt, but live blog and sales-brief drafts could still miss
the requested `preferred_pov=second_person`. Those drafts were persisted with
`brand_voice_audit.passed=false`, which means the UI/review flow can show a
selected brand voice while the generated asset did not actually honor it.

This slice promotes failed brand-voice audit warnings into generation blockers
for the two surfaces that missed in the live proof: blog posts and sales
briefs. Blog generation already has a quality-repair loop, so it can repair a
second-person miss before persistence. Sales briefs currently have no repair
loop, so a miss fails visibly and does not save a draft that claims brand voice
was applied.

## Scope (this PR)

Ownership lane: content-ops/gate-a-output-quality
Slice phase: Production hardening

1. Convert failed brand-voice audits into deterministic quality blockers for
   blog posts and sales briefs when a brand voice profile is supplied.
2. Let blog generation use its existing repair loop to fix second-person POV
   misses before persistence.
3. Make sales-brief generation fail visibly on second-person POV misses instead
   of persisting a draft with `brand_voice_audit.passed=false`.
4. Add focused tests for blog repair, blog no-repair blocking, sales-brief
   blocking, and compliant second-person pass-through.
5. Address review feedback by keeping private generation metadata out of the
   brand-voice audit text surface.

### Review Contract

- Acceptance criteria:
  - [ ] A parsed blog post with `preferred_pov=second_person` and no
        `you`/`your` terms is treated as a quality blocker instead of being
        persisted.
  - [ ] Blog quality repair receives a concrete brand-voice repair instruction
        and can persist a repaired second-person draft.
  - [ ] A parsed sales brief with `preferred_pov=second_person` and no
        `you`/`your` terms is skipped with a visible `quality_blocked` error.
  - [ ] A compliant second-person draft for each surface still persists; no
        brand voice means existing behavior is unchanged.
- Affected surfaces: extracted brand-voice helpers, blog generation quality
  checks/repair, sales-brief quality checks, focused extracted tests.
- Risk areas: over-blocking content with no brand voice, disabling existing
  blog quality repair, adding a hard dependency on live LLM behavior.
- Reviewer rules triggered: R1, R2, R10.

### Files touched

- `extracted_content_pipeline/blog_generation.py`
- `extracted_content_pipeline/brand_voice.py`
- `extracted_content_pipeline/sales_brief_generation.py`
- `plans/PR-Gate-A-Brand-Voice-Second-Person-Enforcement.md`
- `tests/test_extracted_blog_generation.py`
- `tests/test_extracted_brand_voice.py`
- `tests/test_extracted_sales_brief_generation.py`

## Mechanism

`brand_voice_result_metadata(...)` already attaches `_brand_voice_audit` to
parsed LLM payloads when a profile is supplied. This PR adds a small helper in
`brand_voice.py` that converts failed audit fields into deterministic blocker
codes:

```text
brand_voice:preferred_pov_second_person_not_detected
brand_voice:banned_term:<term>
```

The audit text flattener ignores underscore-prefixed private metadata fields
such as `_variant_angle`, `_model`, and `_usage`; only user-facing draft fields
can satisfy POV, banned-term, or reading-level checks.

Blog generation appends those blockers to its existing quality blockers. If
quality repair is enabled, the same repair loop asks the LLM to fix the brand
voice miss, then re-runs the quality check before persistence. Sales-brief
generation appends the same blockers to its quality check; because sales
briefs do not have a repair loop yet, a failed brand-voice audit is skipped
with `reason="quality_blocked"` and the blocker list in the error payload.

## Intentional

- This does not add a new LLM repair path to sales briefs. The narrow
  production-hardening requirement is to stop persisting brand-voice misses;
  a sales-brief repair loop is a larger follow-up.
- This does not change the audit detector itself. The live miss was already
  detected by `brand_voice_audit`; the gap was that generators treated the
  failed audit as metadata only.
- Failed brand-voice audits still block when a caller disables the generic
  quality gate. A selected brand voice is a caller-visible contract, so the
  generator must not persist output that its own brand-voice audit says failed.
- Local review caller hints were inspected. Host references only construct the
  blog/sales-brief services with unchanged signatures; the landing/report
  `_quality_check` hits are same-name methods on other generator classes, not
  call sites for the modified blog/sales-brief methods.
- This does not run another live Gate A smoke. The slice is a deterministic
  enforcement change; the messy-ticket rerun remains the later validation
  artifact after the structural quality fixes land.

## Deferred

- Gate A landing-page distinctness: make whole-page variants meaningfully
  different, not only hero-headline variants.
- Gate A blog prose quality: resolve debug-style source narration.
- Gate A messy-ticket rerun: rerun the live proof on noisy support-ticket data
  after structural and prompt-quality fixes are in place.
- Optional sales-brief brand-voice repair loop: add a repair attempt path if
  product wants automatic repair instead of fail-visible skip for sales briefs.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_extracted_brand_voice.py tests/test_extracted_blog_generation.py tests/test_extracted_sales_brief_generation.py -q`
  - PASS (`122 passed` after review fix).
- `bash scripts/validate_extracted_content_pipeline.sh` - PASS.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - PASS.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - PASS (`Atlas runtime import findings: 0`).
- `bash scripts/check_ascii_python.sh` - PASS.
- `bash scripts/run_extracted_pipeline_checks.sh` - PASS (`extracted_reasoning_core`: `295 passed`; `extracted_content_pipeline`: `3268 passed, 10 skipped`; one `pynvml` warning).
- `bash scripts/local_pr_review.sh --current-pr-body-file tmp/gate-a-brand-voice-second-person-enforcement-pr-body.md` - PASS; caller hints inspected as noted in Intentional.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/blog_generation.py` | 50 |
| `extracted_content_pipeline/brand_voice.py` | 25 |
| `extracted_content_pipeline/sales_brief_generation.py` | 13 |
| `plans/PR-Gate-A-Brand-Voice-Second-Person-Enforcement.md` | 138 |
| `tests/test_extracted_blog_generation.py` | 75 |
| `tests/test_extracted_brand_voice.py` | 44 |
| `tests/test_extracted_sales_brief_generation.py` | 54 |
| **Total** | **399** |

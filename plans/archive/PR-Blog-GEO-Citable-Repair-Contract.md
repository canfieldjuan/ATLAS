# PR: Blog GEO Citable Repair Contract

## Why this slice exists

After save-time and export GEO citable-section validation were aligned, the
36-row SaaS demo support-ticket blog no longer false-saves. It now correctly
fails before save on `geo_citable_section_structure_missing` after two repair
attempts.

The remaining source gap is the repair contract. The current repair instruction
asks for two independently citable H2 sections, but it does not make the exact
shape mechanical enough for the model: the first paragraph immediately after
each of at least two H2 headings must be 40-120 words and must include the exact
target keyword or topic term the gate checks.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-provider
Slice phase: Production hardening

1. Promote the parked citable-section repair-contract item from `HARDENING.md`.
2. Tighten the `geo_citable_section_structure_missing` repair guidance so it
   tells the model to rewrite at least two H2 sections in the exact shape the
   save-time gate requires.
3. Pin the repair guidance with focused assertions.
4. Do not run another live Haiku retry in this source-fix slice.

### Files touched

- `plans/PR-Blog-GEO-Citable-Repair-Contract.md` - Plan doc for this source fix.
- `HARDENING.md` - Remove the promoted citable-section repair-contract item.
- `extracted_content_pipeline/blog_generation.py` - Strengthen citable-section repair guidance.
- `tests/test_extracted_blog_generation.py` - Assert the repair prompt names the exact citable-section shape.

## Mechanism

`_blog_quality_repair_guidance` maps quality blocker codes into LLM repair
instructions. This PR keeps the existing citable-section blocker mapping, but
makes it explicit that repair must:

- rewrite at least two H2 sections;
- make the first paragraph immediately after each H2 40-120 words;
- include the exact `target_keyword` from the previous JSON in both opening
  paragraphs, or the exact named topic term if no target keyword exists;
- avoid relying on the title, intro, blockquotes, bullets, or later paragraphs
  to satisfy the citable-section check.

That matches the save-time gate's citable-section scope and gives the repair
loop a mechanical target.

## Intentional

- This does not weaken the save-time gate or export readiness.
- This does not run another live model retry; the next validation slice should
  rerun the 36-row SaaS demo blog path after this source fix lands.
- This does not touch FAQ article generation or FAQ report ownership.

## Deferred

- Future PR: rerun the 36-row SaaS demo blog path with Haiku and accept the
  fixture only if it saves, support-ticket evaluation passes, and exported
  SEO/AEO plus GEO readiness are ready.
- Future PR: add a scripted regression gate after an accepted SaaS demo blog
  fixture lands.
- Parked hardening:
  - LLM usage storage schema mismatch hides per-run cost telemetry.

## Verification

- python -m pytest tests/test_extracted_blog_generation.py -q - passed, 68 tests.
- python -m py_compile extracted_content_pipeline/blog_generation.py tests/test_extracted_blog_generation.py - passed.
- bash scripts/validate_extracted_content_pipeline.sh - passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline - passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt - passed.
- bash scripts/check_ascii_python.sh - passed.
- Local PR review with the blog GEO citable repair contract PR body - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~85 |
| Repair guidance | ~8 |
| Test | ~5 |
| HARDENING cleanup | ~9 |
| **Total** | **~107** |

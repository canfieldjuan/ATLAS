# PR: Blog GEO Citable Readiness Alignment

## Why this slice exists

The 36-row SaaS demo support-ticket blog retry saved a draft and passed the
deterministic support-ticket truthfulness evaluator, but the exported draft was
still not acceptable: `geo_readiness.status` was `needs_review` because
`citable_section_structure` was missing.

The source issue is a validator mismatch. Save-time blog quality uses
`extracted_quality_gate.blog_pack` and currently lets a citable section pass
when a 40-120 word opening paragraph contains a visible capitalized entity,
even if it does not contain the topic terms or target keyword. Export readiness
uses `extracted_content_pipeline.blog_post_export` and requires the topic terms
when they are available. That means a blog can save as quality-passed and then
fail the exported GEO/AEO/SEO acceptance bar.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-provider
Slice phase: Production hardening

1. Promote the parked citable-section mismatch from `HARDENING.md`.
2. Align save-time GEO citable-section validation with export readiness by
   requiring topic terms when topic terms exist.
3. Add a focused save-time quality-gate regression for visible-entity-only H2
   sections.
4. Do not run another live Haiku retry in this source-fix slice.

### Files touched

- `plans/PR-Blog-GEO-Citable-Readiness-Alignment.md` - Plan doc for this source fix.
- `HARDENING.md` - Remove the promoted validator mismatch item.
- `extracted_quality_gate/blog_pack.py` - Align save-time GEO citable-section logic with export readiness.
- `tests/test_extracted_quality_gate_blog_pack.py` - Pin the stricter citable-section behavior.
- `tests/test_extracted_blog_generation.py` - Update the shared valid blog fixture so generator tests satisfy the stricter save-time citable-section contract.

## Mechanism

`_geo_self_contained_section` is the save-time helper behind
`geo_citable_section_structure_missing`. This PR changes it to match export
readiness:

- keep the 40-120 word first-paragraph requirement;
- if `topic_terms` exist, require at least one topic term in the H2 heading or
  first paragraph;
- only use the visible-entity fallback when no topic terms exist.

That makes save-time validation at least as strict as the exported readiness
contract for the same citable-section signal, so the generation repair loop can
fix the issue before a draft is saved.

## Intentional

- This changes the save-time gate rather than weakening export readiness. The
  product acceptance bar is the exported SEO/AEO/GEO readiness output.
- This does not touch prompt copy or repair guidance. After this lands, the
  existing repair loop should receive `geo_citable_section_structure_missing`
  for this class of draft.
- This does not run a live model retry; the next validation slice should do that
  after the source gate alignment lands.

## Deferred

- Future PR: rerun the 36-row SaaS demo blog path with Haiku and accept the
  fixture only if it saves, generated-content evaluation passes, and exported
  SEO/AEO plus GEO readiness are ready.
- Parked hardening:
  - LLM usage storage schema mismatch hides per-run cost telemetry.

## Verification

- python -m pytest tests/test_extracted_quality_gate_blog_pack.py -q - passed, 43 tests.
- python -m pytest tests/test_extracted_quality_gate_blog_pack.py tests/test_extracted_blog_generation.py tests/test_extracted_blog_post_export.py -q - passed, 120 tests.
- bash scripts/validate_extracted_content_pipeline.sh - passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline - passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt - passed.
- bash scripts/check_ascii_python.sh - passed.
- python -m py_compile extracted_quality_gate/blog_pack.py tests/test_extracted_quality_gate_blog_pack.py tests/test_extracted_blog_generation.py - passed.
- Local PR review with the blog GEO citable readiness alignment PR body - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~80 |
| Quality gate | ~3 |
| Tests | ~40 |
| HARDENING cleanup | ~9 |
| **Total** | **~132** |

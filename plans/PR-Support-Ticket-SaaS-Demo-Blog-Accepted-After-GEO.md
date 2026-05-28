# PR: Support-Ticket SaaS Demo Blog Accepted After GEO

## Why this slice exists

The 36-row SaaS demo support-ticket blog path has now had the observed source
blockers fixed in prior slices: unsupported outcome language, vague-H2 GEO
repair guidance, and save-time/export citable-section drift. The next proof is
not another prompt or validator patch. It is a live Haiku acceptance retry
through the real Content Ops path.

This slice accepts the blog fixture only if the saved draft satisfies the full
bar: the draft saves, deterministic support-ticket generated-content evaluation
passes, exported SEO/AEO readiness is ready, and exported GEO readiness is
ready.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-provider
Slice phase: Functional validation

1. Run one live Haiku blog-post smoke against the 36-row SaaS demo
   support-ticket CSV with saved draft export and generated-content evaluation
   enabled.
2. Commit the exported blog draft as the current accepted SaaS demo blog fixture
   only if all acceptance signals pass.
3. Update the SaaS demo generated-content validation report with the exact live
   run result.
4. If the run still fails, do not patch another source issue in this PR; update
   the report and park the next blocker instead.

### Files touched

- `plans/PR-Support-Ticket-SaaS-Demo-Blog-Accepted-After-GEO.md` - Plan doc for this live acceptance retry.
- `docs/extraction/validation/support_ticket_saas_demo_generated_content_acceptance_2026-05-28.md` - Update SaaS demo blog acceptance status and verification.
- `HARDENING.md` - Park the next source blocker exposed by the live retry.

## Mechanism

The live smoke command uses the existing host harness with the same inputs as
the prior retries:

python scripts/smoke_content_ops_live_generation.py --output blog_post
--support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv
--evaluate-generated-content --export-saved-draft <tmp export path>

The command loads the Atlas env files plus the Haiku override env file, packages
the CSV through the support-ticket input provider, seeds a blog blueprint, calls
the pipeline-routed LLM, saves the draft, exports the saved draft, and runs the
support-ticket generated-content evaluator against the export.

## Intentional

- This uses Haiku for validation spend, not Sonnet.
- This does not change prompts, detectors, repair guidance, or readiness
  validators. The purpose is to prove the merged source fixes end to end.
- This does not touch FAQ article generation or FAQ report ownership.

## Deferred

- Future PR: strengthen the blog citable-section repair contract so the repair
  loop produces at least two 40-120 word H2 opening paragraphs with the exact
  target keyword or topic term.
- Future PR: rerun the 36-row SaaS demo blog path with Haiku and accept the
  fixture only if it saves, support-ticket evaluation passes, and exported
  SEO/AEO plus GEO readiness are ready.
- Future PR: add a scripted regression gate after the accepted SaaS demo blog
  fixture lands.
- Parked hardening:
  - Support-ticket SaaS demo blog still fails the GEO citable-section gate after
    repair.
  - LLM usage storage schema mismatch hides per-run cost telemetry.

## Verification

- Live Haiku blog smoke against the 36-row SaaS demo CSV - failed before save
  after two repair attempts on geo_citable_section_structure_missing; no saved
  draft export was produced and generated-content evaluation did not run.
- Local PR review with the SaaS demo blog accepted-after-GEO PR body - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~80 |
| Validation report update | ~40 |
| HARDENING entry | ~9 |
| **Total** | **~129** |

# PR: Blog GEO Entity Prompt Alignment

## Why this slice exists

The custom blog live smoke proved the seeded blueprint path, LLM routing, and
draft persistence path, but the draft failed the blog quality gate with
`geo_entity_clarity_missing`. The generated article was coherent, but it did
not mention the exact target keyword or named subject early enough for the
existing GEO entity-clarity validator, which checks the title plus the opening
body window.

This slice aligns the blog-generation and quality-repair path with the existing
validator contract instead of weakening the gate.

## Scope (this PR)

Ownership lane: content-ops/blog-geo-prompt-alignment

1. Require exact `target_keyword` / subject clarity in the blog prompt.
2. Add targeted repair guidance for observed SEO/AEO/GEO blocker codes.
3. Decouple blog quality-repair attempts from parse-retry attempts.
4. Normalize overlong SEO titles before validation.
5. Thread blog repair-attempt count through plan and executor config.
6. Sync the extracted Content Ops skill copy and add focused tests.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Blog-GEO-Entity-Prompt-Alignment.md` | Plan doc for this slice. |
| `atlas_brain/skills/digest/blog_post_generation.md` | Source prompt guidance for early subject clarity. |
| `extracted_content_pipeline/skills/digest/blog_post_generation.md` | Synced extracted prompt copy. |
| `extracted_content_pipeline/blog_generation.py` | Targeted quality-repair guidance, SEO title normalization, and multiple repair attempts for gate blocker codes. |
| `extracted_content_pipeline/generation_plan.py` | Include blog repair-attempt count in the step config. |
| `extracted_content_pipeline/content_ops_execution.py` | Pass blog repair-attempt count into the service. |
| `tests/test_atlas_content_ops_infrastructure.py` | Regression coverage for the host-shipped prompt text. |
| `tests/test_extracted_blog_generation.py` | Regression coverage for the quality-repair prompt guidance. |
| `tests/test_extracted_content_generation_plan.py` | Regression coverage for the blog step config. |
| `tests/test_extracted_content_ops_execution.py` | Regression coverage for executor dispatch wiring. |

## Mechanism

The blog GEO validator checks the display title plus opening body window for
topic terms from `target_keyword` and blueprint context. The prompt now tells
the model to put the exact phrase there, and the real-disk skill-store test pins
that instruction.

The repair prompt expands known blocker codes into concrete edits while keeping
the raw blocker list for auditability. Blog generation now has its own
`quality_repair_attempts` config, emits it in the plan, and passes it through
execution so a blocker introduced by the first repair can receive a second
repair. Mechanical SEO-title length misses are trimmed before validation;
content-level blockers stay in the LLM repair path.

## Intentional

- No quality-gate code change. The gate is doing the right source-level check;
  the generation and repair prompts were underspecified.
- No live Sonnet test. Future live validation will use the Haiku override file
  so testing does not spend against the expensive model family.
- No broad prompt rewrite. This is a narrow alignment with the failures observed
  in the custom blog smoke.
- The blog repair-attempt default is 2. That gives the service one follow-up
  after the initial repair without opening an unbounded retry loop.
- SEO title normalization is deterministic because the quality gate already
  enforces a hard character maximum; this avoids spending an LLM repair attempt
  on a one-character title miss.

## Deferred

- Haiku live smokes on this branch exposed the blocker sequence fixed here:
  short content / overlong SEO title / missing entity clarity, then missing
  citable H2 structure, then citation-safety failure after the one-attempt loop.
  The final live run still did not save a draft; further debugging should
  inspect the failed candidate body instead of continuing blind prompt
  iteration.
- Parked hardening: existing `ATLAS-HARDENING.md` items are for the older
  deep-dive blog lane and do not touch this Content Ops blog prompt/readiness
  contract. They remain parked.

## Verification

- Haiku live smokes with `/tmp/atlas-haiku-override.env` -> no saved draft yet; blockers progressed through the prompt/repair gaps listed in Deferred.
- pytest `tests/test_extracted_blog_generation.py` `tests/test_atlas_content_ops_infrastructure.py` `tests/test_extracted_content_generation_plan.py` `tests/test_extracted_content_ops_execution.py` -q -> 136 passed.
- bash `extracted/_shared/scripts/sync_extracted.sh` extracted_content_pipeline -> refreshed 43 mapped files.
- bash `scripts/validate_extracted_content_pipeline.sh` -> passed.
- python `extracted/_shared/scripts/forbid_atlas_reasoning_imports.py` extracted_content_pipeline -> passed.
- python `scripts/audit_extracted_standalone.py` --fail-on-debt -> passed.
- bash `scripts/check_ascii_python.sh` -> passed.
- bash `scripts/local_pr_review.sh` -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~105 |
| Prompt sync | ~8 |
| Repair prompt/loop | ~175 |
| Plan/dispatch wiring | ~6 |
| Tests | ~155 |
| **Total** | **~449** |

This is over the 400 LOC soft cap because the live-smoke repair path and PR
review both exposed source-level integration issues in the same loop:
prompt/gate alignment, bounded multi-attempt repair, deterministic metadata
normalization, dispatch wiring, and debuggable repair-failure attribution.

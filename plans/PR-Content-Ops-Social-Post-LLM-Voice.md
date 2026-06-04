# PR-Content-Ops-Social-Post-LLM-Voice

## Why this slice exists

PR-Content-Ops-Brand-Voice-Profile deliberately skipped social-post voice
support because `social_post` was still a deterministic template output. The
subsequent storage, editor, onboarding, scrape, and preset slices made saved
brand voice usable from the Content Ops UI, and each deferred
`PR-Content-Ops-Social-Post-LLM-Voice` as the remaining product path.

Today social posts are persisted and reviewable, but selecting a brand voice
profile still does not affect them. This slice adds the thinnest LLM-backed
social-post variant over the existing source-evidence drafts so the selected
brand voice is actually consumed instead of silently ignored.

This is over the 400 LOC soft cap because the first executable social-post
voice path needs the prompt, synced manifest entry, generator behavior,
dispatcher/plan wiring, host service wiring, and fail-closed tests together.
Splitting before the host wiring would leave the new LLM path unreachable;
splitting before the negative tests would make the "selected brand voice is not
silently ignored" claim unenforced.

## Scope (this PR)

Ownership lane: content-ops/brand-voice/social-post-llm
Slice phase: Vertical slice

1. Keep the existing deterministic social-post generator as the fallback path.
2. Add optional LLM social-post rewriting when a request supplies brand voice;
   no brand voice means the existing deterministic behavior remains unchanged.
3. Fail the social-post step when brand voice is supplied but the service has no
   LLM client or prompt, so a selected profile is not silently ignored.
4. Thread `brand_voice` through the social-post dispatcher and include
   `brand_voice_profile_id` in social-post plan metadata.
5. Wire the host DB-backed social-post service with the existing configured
   Content Ops LLM and skill store while preserving the no-LLM deterministic
   singleton.
6. Add a host-owned social-post prompt synced into the extracted package by the
   manifest.
7. Add focused package and host wiring tests for LLM rewrite, fail-closed
   missing LLM/prompt behavior, plan metadata, dispatcher threading, and
   deterministic fallback.

### Files touched

- `atlas_brain/_content_ops_services.py`
- `atlas_brain/skills/digest/social_post_generation.md`
- `extracted_content_pipeline/content_ops_execution.py`
- `extracted_content_pipeline/generation_plan.py`
- `extracted_content_pipeline/manifest.json`
- `extracted_content_pipeline/skills/digest/social_post_generation.md`
- `extracted_content_pipeline/social_post_generation.py`
- `plans/PR-Content-Ops-Social-Post-LLM-Voice.md`
- `tests/test_atlas_content_ops_execution_services.py`
- `tests/test_extracted_content_generation_plan.py`
- `tests/test_extracted_content_ops_execution.py`
- `tests/test_extracted_social_post_generation.py`

## Mechanism

`SocialPostGenerationService` gains optional `llm` and `skills` ports using the
same `LLMClient` / `SkillStore` contracts as campaign, blog, landing page, and
sales brief. It first builds the existing evidence-backed deterministic posts.
When no brand voice is supplied, it returns those posts exactly as before.

When brand voice is supplied, the service resolves it through
`brand_voice_profile_from_mapping(...)`, loads `digest/social_post_generation`,
injects the standard brand-voice block with
`apply_brand_voice_to_system_prompt(...)`, and asks the configured Content Ops
LLM to rewrite each deterministic post into JSON:

```json
{"channel": "linkedin", "text": "bounded social copy"}
```

The rewrite prompt carries the original post and source evidence; the model is
allowed to change wording and rhythm only. Parsed LLM posts keep the existing
fields (`source_id`, `vendor_name`, `pain_points`, etc.) and add
`_brand_voice_profile` / `_brand_voice_audit` metadata before persistence. If a
brand voice request reaches a service without `llm` or `skills`, or without the
prompt, the step fails rather than returning unvoiced deterministic copy.

Host wiring changes only the DB-enabled social-post builder: when an LLM and
pool are available it passes the existing OpenRouter-configured Content Ops LLM
client and skill store into `SocialPostGenerationService`. The no-DB/no-LLM
singleton remains deterministic and wired.

## Intentional

- This does not add a new output id. `social_post` remains the product output;
  brand voice switches it to the LLM rewrite path only when the request has a
  resolved profile.
- Deterministic social posts still run without LLM or DB services. The fail
  closed behavior applies only when brand voice is selected.
- No local-model route is introduced. Host LLM wiring reuses the existing
  Content Ops OpenRouter path with `auto_activate_ollama=False`.
- No social-post edit/repair UI in this slice. The existing generated-asset
  review table remains the approval surface.

## Deferred

- `PR-Content-Ops-Social-Post-Channel-Variants`: platform-specific variants
  beyond the current single LinkedIn draft.
- `PR-Content-Ops-Brand-Voice-Settings-Page`: optional standalone management
  page if the inline New Run panel becomes too dense.

Parked hardening: none.

## Verification

- `python -m py_compile extracted_content_pipeline/social_post_generation.py extracted_content_pipeline/generation_plan.py extracted_content_pipeline/content_ops_execution.py atlas_brain/_content_ops_services.py tests/test_extracted_social_post_generation.py tests/test_extracted_content_generation_plan.py tests/test_extracted_content_ops_execution.py tests/test_atlas_content_ops_execution_services.py`
  -- passed.
- `pytest tests/test_extracted_social_post_generation.py tests/test_extracted_content_generation_plan.py tests/test_extracted_content_ops_execution.py -q`
  -- 126 passed.
- `pytest tests/test_atlas_content_ops_execution_services.py -q` -- 28 passed.
- `bash scripts/validate_extracted_content_pipeline.sh` -- passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline`
  -- passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` -- 0 findings.
- `bash scripts/check_ascii_python.sh` -- passed.
- `bash extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline` --
  refreshed 46 mapped files.
- `bash scripts/run_extracted_pipeline_checks.sh` -- 295 reasoning-core tests
  passed; 3115 extracted-content tests passed / 10 skipped / 1 warning.
- `git diff --check` -- passed.
- `bash scripts/local_pr_review.sh --current-pr-body-file tmp/pr-body-content-ops-social-post-llm-voice.md`
  -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/_content_ops_services.py` | 20 |
| `atlas_brain/skills/digest/social_post_generation.md` | 59 |
| `extracted_content_pipeline/content_ops_execution.py` | 25 |
| `extracted_content_pipeline/generation_plan.py` | 6 |
| `extracted_content_pipeline/manifest.json` | 4 |
| `extracted_content_pipeline/skills/digest/social_post_generation.md` | 59 |
| `extracted_content_pipeline/social_post_generation.py` | 303 |
| `plans/PR-Content-Ops-Social-Post-LLM-Voice.md` | 145 |
| `tests/test_atlas_content_ops_execution_services.py` | 132 |
| `tests/test_extracted_content_generation_plan.py` | 9 |
| `tests/test_extracted_content_ops_execution.py` | 43 |
| `tests/test_extracted_social_post_generation.py` | 178 |
| **Total** | **983** |

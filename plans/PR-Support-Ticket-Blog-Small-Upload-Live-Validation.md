# PR: Support-Ticket Blog Small Upload Live Validation

## Why this slice exists

PR #987 tightened the support-ticket blog prompt so small no-outcome/no-resolution
uploads should produce compact, non-repetitive descriptive drafts instead of
long articles. The deterministic prompt-contract test proved the instruction is
present, but we still need one live Haiku proof that the real route now behaves
better on the same 4-row packaged support-ticket CSV that previously produced a
truth-safe but 10k-output-token draft.

This slice validates the product-polish fix and, if live output shows the trim
is still too loose, tightens the small-upload prompt contract at the source
before rerunning the live proof.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-provider-blog-polish
Slice phase: Functional validation

1. Run the live support-ticket blog smoke through the host DB pool, packaged
   skill prompt, pipeline-routed Haiku model, save path, export path, and
   deterministic generated-content evaluator.
2. If the run stays truth-safe but misses the compact/GEO shape, tighten the
   small-upload support-ticket prompt contract in the host and extracted prompt
   copies.
3. Make the blog quality policy honor the same compact-word-count exception
   only for small no-outcome/no-resolution support-ticket contexts.
4. Pin the prompt and quality-policy contracts with focused tests.
5. Export the saved draft/result artifacts under `tmp/`.
6. Rerun the deterministic support-ticket generated-content evaluator against
   the saved draft export.
7. Manually inspect the saved draft for:
   - compact article shape relative to the previous 10k-output-token result
   - no repeated H2 sections covering the same cluster or FAQ gap
   - no unsupported outcome, cadence, date-window, answer-step, or product-step
     claims
8. Record the command, artifact paths, usage, length/section summary, evaluator
   result, and manual audit in a validation doc.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Support-Ticket-Blog-Small-Upload-Live-Validation.md` | Plan doc for the live validation slice. |
| `extracted_content_pipeline/blog_generation.py` | Apply the compact support-ticket word-count policy before the blog quality gate. |
| `atlas_brain/skills/digest/blog_post_generation.md` | Tighten small-upload support-ticket prompt shape if live proof shows the prior trim is too loose. |
| `extracted_content_pipeline/skills/digest/blog_post_generation.md` | Keep extracted skill prompt aligned with the host copy. |
| `tests/test_atlas_content_ops_infrastructure.py` | Pin the tightened small-upload support-ticket prompt contract. |
| `tests/test_extracted_blog_generation.py` | Pin compact support-ticket blog quality-policy selection and repair guidance. |
| `docs/extraction/validation/support_ticket_blog_small_upload_live_validation_2026-05-26.md` | Live Haiku validation record. |

## Mechanism

Use the existing live smoke harness with the Haiku override and the packaged
support-ticket CSV:

```bash
python scripts/smoke_content_ops_live_generation.py \
  --output blog_post \
  --account-id acct_support_ticket_blog_small_upload_live_validation_20260526 \
  --support-ticket-csv \
  --env-file /home/juan-canfield/Desktop/Atlas/.env \
  --env-file /home/juan-canfield/Desktop/Atlas/.env.local \
  --env-file tmp/support_ticket_live_haiku_eval_20260525/haiku.env \
  --export-saved-draft tmp/support_ticket_blog_small_upload_live_validation_20260526/blog-post-draft.json \
  --output-result tmp/support_ticket_blog_small_upload_live_validation_20260526/blog-post-result.json \
  --evaluate-generated-content \
  --json
```

Then rerun:

```bash
python scripts/evaluate_support_ticket_generated_content.py \
  --output blog_post \
  tmp/support_ticket_blog_small_upload_live_validation_20260526/blog-post-draft.json \
  --pretty
```

The artifacts remain uncommitted in `tmp/`; the committed validation doc records
the evidence.

The first live run after #987 is allowed to fail the compact/GEO shape because
that is what this slice is validating. If it does, the source prompt will be
tightened to make small uploads use a short FAQ review brief shape: 700-1100
words, 3-4 H2 sections, no H3 subsections, no broad scaling/process section, and
first H2 openings that preserve the exact `target_keyword`/support-ticket
subject required by the GEO citable-section checker. The blog quality gate will
also receive a matching `min_words=700` / `target_words=1100` policy for that
same small-upload context, while every other blog context keeps the default
1500-word floor. The final validation run is the result that must pass the
slice.

## Intentional

- This keeps prompt and quality-policy tuning scoped to the exact
  live-validation failure. If live output is truth-unsafe, the slice will stop
  and fix that blocker before any merge.
- Haiku remains the model for this check because it is the cheap model we use
  for validation and the one that previously amplified the long/repetitive
  output.
- This does not implement customer-language keyword promotion or standalone FAQ
  Article output; those remain future product slices coordinated with the FAQ
  lane.
- Cross-layer caller hints for `BlogPostGenerationService` are covered by the
  live host Content Ops route proof and focused service tests. The
  `_positive_int_context` hint is a same-name private-helper collision with
  `content_ops_execution.py`; this PR does not change that module's helper.

## Deferred

- If the live draft is still truth-safe but cosmetically imperfect, non-blocking
  polish will be parked in `ATLAS-HARDENING.md` rather than expanding this
  validation PR.
- Broader acceptance testing across many customer CSV shapes remains a later
  robust-testing slice.
- Parked hardening: none planned.

## Verification

- Initial live Haiku blog-post smoke with support-ticket CSV and
  generated-content evaluation - passed source-truthfulness but missed compact
  shape/GEO readiness.
- Second live Haiku blog-post smoke after prompt tightening - blocked before
  save on the old 1500-word gate and an unsupported "find answers without
  opening a support ticket" claim.
- Final live Haiku blog-post smoke after compact quality-policy wiring - passed;
  saved draft `4dc73f34-3bfa-45f1-91ab-afe19a9df339`.
- Explicit generated-content evaluator rerun against the saved draft export -
  passed.
- Manual copy audit for length, repeated sections, unsupported outcomes,
  unsupported timeframes/cadence, and concrete answer steps without resolution
  evidence - passed.
- Command: python -m pytest tests/test_atlas_content_ops_infrastructure.py -q
  - `12 passed`
- Command: python -m pytest tests/test_extracted_blog_generation.py tests/test_atlas_content_ops_infrastructure.py -q
  - `73 passed`
- Command: python -m pytest tests/test_extracted_blog_generation.py tests/test_evaluate_support_ticket_generated_content.py tests/test_atlas_content_ops_infrastructure.py -q
  - `115 passed`
- Extracted package guardrails for the synced prompt file:
  - Command: bash scripts/validate_extracted_content_pipeline.sh - passed
  - Command: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline - passed
  - Command: python scripts/audit_extracted_standalone.py --fail-on-debt - passed
  - Command: bash scripts/check_ascii_python.sh - passed
  - Command: bash extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline - passed
- Command: bash scripts/local_pr_review.sh --current-pr-body-file tmp/pr-support-ticket-blog-small-upload-live-validation-body.md
  - passed

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~160 |
| Prompt/test updates | ~45 |
| Quality-policy wiring | ~90 |
| Validation doc | ~170 |
| **Total** | **~500** |

This exceeds the 400 LOC soft cap because the slice had to carry the live
validation record plus the source fix exposed by that validation: compact prompt
shape and matching quality-policy selection must land together or the live path
continues to block compact drafts.

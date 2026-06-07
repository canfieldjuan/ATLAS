# PR-Gate-A-Blog-Variant-Persistence

## Why this slice exists

PR-Gate-A-Live-Output-Quality-Proof proved the live Gate A path on real
Postgres and the configured cloud/OpenRouter route, but the run failed a
structural product requirement: two successful `blog_post` variants both
returned the same persisted draft id and the exact export contained only one
approved blog row. That means `variant_count > 1` can still collapse in the
review/export surface, so marketers do not get distinct blog drafts to compare
or approve.

This slice promotes the matching `HARDENING.md` item into scope and fixes only
that persistence failure. The other Gate A quality failures remain parked
because they need prompt/evaluator/product-quality work, not draft identity
hardening.

## Scope (this PR)

Ownership lane: content-ops/gate-a-blog-variant-persistence
Slice phase: Production hardening

1. Make blog-post variant generation persist reviewable drafts under distinct,
   deterministic slugs when `variant_angle` is present.
2. Preserve the normal non-variant behavior where regenerating the same blog
   slug updates the existing draft instead of creating duplicates.
3. Add regression coverage proving variant angles with the same model-returned
   slug no longer collapse before the repository upsert.
4. Remove the resolved blog-variant persistence item from `HARDENING.md`.

### Review Contract

- Acceptance criteria:
  - [ ] A blog generation call with `variant_angle` appends a stable
        human-readable variant suffix to the draft slug.
  - [ ] Two variant generations for the same blueprint/model slug produce
        distinct saved draft slugs, so the repository can return distinct ids.
  - [ ] A non-variant generation keeps the existing slug unchanged.
  - [ ] The existing `metadata.variant_angle` threading and LLM metadata remain
        unchanged.
  - [ ] The repository upsert contract is not weakened: same-tenant,
        same-slug non-variant reruns still update rather than duplicate.
- Affected surfaces: extracted blog generation, blog draft persistence keys,
  generated-asset review/export identity indirectly.
- Risk areas: backward compatibility of slugs, idempotency, variant slug
  determinism, scope creep into unrelated Gate A quality fixes.
- Reviewer rules triggered: R1, R2, R5, R8, R10.

### Files touched

- `HARDENING.md`
- `extracted_content_pipeline/blog_generation.py`
- `plans/PR-Gate-A-Blog-Variant-Persistence.md`
- `tests/test_extracted_blog_generation.py`

## Mechanism

`BlogPostGenerationService.generate(...)` already threads
`variant_angle` into the prompt, LLM metadata, parsed payload, and draft
metadata. The collapse happens later because `_build_draft(...)` still uses the
model/blueprint slug as-is, and `PostgresBlogPostRepository.save_drafts(...)`
upserts on `blog_posts.slug`.

The fix keeps the repository contract intact and changes only the generated
variant draft key:

```python
base_slug = _slugify(parsed.get("slug") or blueprint.get("slug") or title)
slug = _slug_with_variant_suffix(base_slug, variant_angle)
```

The suffix is derived deterministically from the leading variant label
(`"Pain-led: ..."` -> `pain-led`, `"Outcome-led: ..."` -> `outcome-led`) and is
appended only when `variant_angle` is present and the slug does not already end
with that suffix. The helper preserves the existing 100-character slug cap by
reserving room for `-<suffix>`.

## Intentional

- No database migration: the existing unique slug upsert remains the
  persistence guard.
- No repository rewrite: changing `ON CONFLICT (slug)` would risk the
  non-variant idempotency contract. Variant identity is a generation concern.
- No live Gate A rerun in this PR: this closes the structural duplicate-id
  failure with focused regression tests; the next acceptance run should happen
  after the remaining quality hardening slices.
- No prompt-quality fixes: debug-style blog prose, brand-voice misses,
  landing-page sameness, sales-brief type drift, and messy-ticket grounding are
  separate product-quality failures.
- Cross-layer caller hints were inspected. Host wiring references are
  unaffected because the `BlogPostGenerationService` constructor and
  `generate(...)` port are unchanged; content-ops execution coverage ran in the
  full extracted check. Same-name `_build_draft`, `_service`, and test fake
  `save_drafts` hints outside this module are regex false positives, not shared
  call sites.

## Deferred

- Gate A blog prose quality: resolve the parked "debug-style source narration"
  item before rerunning the acceptance gate.
- Gate A brand voice enforcement: resolve the parked second-person miss.
- Gate A sales brief mode fidelity: keep the requested `brief_type` from being
  overwritten by model JSON.
- Gate A landing-page distinctness: make whole-page variants meaningfully
  different, not only hero-headline variants.
- Gate A messy-ticket rerun: rerun the live proof on noisy support-ticket data
  after the structural and quality fixes are in place.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_extracted_blog_generation.py -q`: 78 passed.
- `bash scripts/validate_extracted_content_pipeline.sh`: passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline`: passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt`: passed.
- `bash scripts/check_ascii_python.sh`: passed.
- `bash scripts/run_extracted_pipeline_checks.sh`: 3249 passed, 10 skipped.
- Pending before push: `bash scripts/local_pr_review.sh --current-pr-body-file <body-file>`.

## Estimated diff size

| File | LOC |
|---|---:|
| `HARDENING.md` | 9 |
| `extracted_content_pipeline/blog_generation.py` | 24 |
| `plans/PR-Gate-A-Blog-Variant-Persistence.md` | 129 |
| `tests/test_extracted_blog_generation.py` | 47 |
| **Total** | **209** |

# PR-Landing-Page-Quality-Gate-Readiness

## Why this slice exists

PR #710 added landing-page SEO/AEO/GEO readiness summaries to review/export
rows. That gives operators visibility, but the generator can still persist
drafts with malformed slugs, placeholder CTA URLs, unresolved template tokens,
or metadata that clearly does not match the visible page.

This slice moves the safest draft-level checks into the landing-page quality
pack. The goal is not to make every readiness issue a blocker. The quality gate
should block structural or unsafe draft defects and warn on weaker SEO/GEO
signals that a reviewer can still fix.

## Scope (this PR)

1. Block malformed or generic landing-page slugs.
2. Block placeholder or JavaScript primary CTA URLs.
3. Block unresolved template placeholders across visible page surfaces.
4. Warn when SEO title metadata is missing or too long.
5. Warn when section titles are too generic to describe the offer.
6. Warn when metadata does not share meaningful terms with visible page copy.
7. Add focused quality-pack tests and run landing-page generation/export
   integration coverage.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Landing-Page-Quality-Gate-Readiness.md` | Plan doc for this implementation slice. |
| `extracted_quality_gate/landing_page_pack.py` | Add selected landing-page readiness validators to the quality gate. |
| `tests/test_extracted_quality_gate_landing_page_pack.py` | Cover new blocker and warning behavior. |

## Mechanism

The pack remains a pure deterministic validator. It still receives the
structured landing-page payload through `QualityInput.context` and returns a
`QualityReport` with blockers, warnings, score, and decision metadata.

New blockers:

- `invalid_slug`
- `placeholder_cta_url`
- `unresolved_placeholder`

New warnings:

- `generic_section_title`
- `missing_meta_title_tag`
- `meta_title_tag_too_long`
- `metadata_inconsistent`

These checks are intentionally narrower than the full review/export readiness
summary. They cover defects that are clearly detectable before save without
requiring publish-time context, public URL routing, structured data, analytics,
or evidence-store lookups.

## Intentional

- No prompt changes.
- No export/API changes.
- No generated-asset UI changes.
- No database schema changes.
- No publish-level crawler, structured-data, canonical, or rendered HTML
  verification.
- No hard block for every SEO/AEO/GEO readiness miss. Drafts can still save
  with warnings when the issue is reviewable copy quality rather than a broken
  artifact.

## Deferred

- Prompt alignment so generated landing pages naturally satisfy the stricter
  quality pack more often.
- Public renderer/publish verification once generated landing pages have a
  concrete hosted route.
- Evidence-aware claim validation that can distinguish supported numeric claims
  from unsupported ones.

## Verification

- `pytest tests/test_extracted_quality_gate_landing_page_pack.py -q` -> passed
  30/30 tests.
- `pytest tests/test_extracted_landing_page_generation.py tests/test_extracted_landing_page_export.py tests/test_extracted_quality_gate_landing_page_pack.py -q`
  -> passed 60/60 tests.
- Python compile command over `extracted_quality_gate/landing_page_pack.py` and
  `tests/test_extracted_quality_gate_landing_page_pack.py` -> passed 2/2
  files.
- `git diff --check` -> passed with 0 whitespace errors.
- `bash scripts/local_pr_review.sh origin/main` -> passed 3/3 top-level
  checks: pre-push audit wrapper, plan/code consistency, and `git diff
  --check`.
- The pre-push audit wrapper inside local review reported all 8 internal checks
  passed: MCP tool counts, MCP port assignments, MCP tool-name inventories,
  extracted manifest sync, plan shape, plan files touched, plan diff size, and
  ASCII Python policy.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~100 |
| Landing-page quality pack | ~215 |
| Tests | ~90 |
| Total | ~405 |

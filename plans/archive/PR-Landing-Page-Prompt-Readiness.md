# PR-Landing-Page-Prompt-Readiness

## Why this slice exists

PR #711 tightened the landing-page quality gate, but the landing-page
generation prompt still allowed output that the gate now warns on or blocks.
The most direct mismatch was `meta.title_tag`: the old prompt told the model to
skip the title tag if it matched the page title, while the quality gate now
warns when the title tag is missing.

This slice aligns the bundled landing-page prompt with the readiness contract
and quality gate so generation is asked for the same structure that review and
validation expect.

## Scope (this PR)

1. Require valid lowercase hyphenated slugs and call out generic slug values to
   avoid.
2. Require concrete CTA URLs and prohibit placeholder or JavaScript URLs.
3. Require `meta.title_tag` and keep metadata aligned with visible page copy.
4. Ask for specific section headings instead of generic labels.
5. Add readiness instructions for first-viewport clarity, problem/solution
   coverage, and objection coverage when context supports it.
6. Add explicit no-fake-proof and no-placeholder instructions.
7. Add prompt-regression tests for the new contract language.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Landing-Page-Prompt-Readiness.md` | Plan doc for this prompt-alignment slice. |
| `extracted_content_pipeline/skills/digest/landing_page_generation.md` | Align the bundled landing-page generation prompt with readiness and quality-gate expectations. |
| `tests/test_extracted_campaign_skill_registry.py` | Add prompt-regression tests for readiness language and fake-proof guardrails. |

## Mechanism

The prompt update keeps the same JSON contract and does not change runtime
parsing, persistence, or quality-gate code. It only narrows field guidance:

- `slug` must be URL-safe and non-generic.
- `cta.url` must be real enough for draft review, not a placeholder.
- `meta.title_tag` must always be present.
- `sections.title` should describe the offer, audience, problem, or buyer
  question.
- proof, reference ids, testimonials, logos, and source ids must not be
  invented when the campaign does not provide them.

The registry tests assert that the packaged prompt keeps these instructions in
place.

## Intentional

- No generator code changes.
- No quality-gate changes.
- No export/API changes.
- No frontend review UI changes.
- No publish-level checks.

## Deferred

- Add repair-loop behavior that feeds quality-gate blocker details back into
  the model for regeneration.
- Add public-renderer publish verification for crawler-visible HTML, metadata,
  canonical URL, structured data, and CTA routes.
- Evaluate whether future prompts should emit a richer FAQ/objection structure
  when campaign context is strong enough.

## Verification

- `pytest tests/test_extracted_campaign_skill_registry.py tests/test_extracted_landing_page_generation.py tests/test_extracted_quality_gate_landing_page_pack.py -q`
  -> passed 66/66 tests.
- Python compile command over `tests/test_extracted_campaign_skill_registry.py`
  -> passed 1/1 file.
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
| Plan | ~95 |
| Prompt | ~20 |
| Tests | ~20 |
| Total | ~135 |

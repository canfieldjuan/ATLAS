# PR-Content-Ops-FAQ-CFPB-Report-Intent

## Why this slice exists

A real 1,000-row FAQ run against the local CFPB archive now passes when the
input is filtered to narrative-bearing rows, but the generated FAQ exposed one
output-quality issue: the generic `reporting friction` intent matches CFPB
complaints that merely mention a report, identity-theft report, or credit
context. That produced a SaaS-style reporting bucket for financial complaints.

This slice narrows the reporting intent so plain `report` does not override
CFPB/financial complaint topics during production-sized FAQ runs.

## Scope (this PR)

1. Remove the standalone `report` keyword from the default reporting-friction
   intent rule.
2. Keep reporting/export/dashboard SaaS cases covered with more specific
   phrases.
3. Add a CFPB-shaped regression test showing identity-theft report language no
   longer becomes reporting friction.
4. Rerun the 1,000-row CFPB narrative smoke and record the outcome.

### Files touched

- `plans/PR-Content-Ops-FAQ-CFPB-Report-Intent.md`
- `extracted_content_pipeline/ticket_faq_markdown.py`
- `tests/test_extracted_ticket_faq_markdown.py`

## Mechanism

`DEFAULT_INTENT_RULES` keeps the `reporting friction` topic, but the keyword set
changes from generic `report` to explicit export/reporting-product phrases such
as `report export`, `dashboard`, `attribution`, and `analytics report`. The
existing keyword matcher remains unchanged.

The regression test uses CFPB-shaped complaint evidence that mentions an
identity-theft report while the issue is opening an account. The expected output
is `opening an account`, not `reporting friction`.

## Intentional

- This is a generator intent-quality fix, not more logging or visibility work.
- No output-check semantics change; the existing pass/fail checks stay stable.
- The run uses a local subset of the public CFPB archive because live CFPB
  availability is external.

## Deferred

- Broader financial-domain intent expansion remains separate if another real
  large-row run exposes a distinct misclassification.
- Warning noise for missing vendor/contact metadata remains separate from FAQ
  generator behavior because it does not affect FAQ output quality.

## Verification

- Focused FAQ Markdown and scale-smoke tests passed, 81 tests.
- 1,000-row local CFPB narrative smoke passed: 1,000 usable source rows,
  12 generated FAQ items, all output checks true. The prior reporting-friction
  bucket disappeared; the affected rows moved into financial/account topics.
- Full extracted pipeline checks passed, including 295 reasoning-core tests and
  1,598 extracted Content Ops tests.
- Local PR review.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | 72 |
| FAQ intent rule | 8 |
| Tests | 20 |
| **Total** | **100** |

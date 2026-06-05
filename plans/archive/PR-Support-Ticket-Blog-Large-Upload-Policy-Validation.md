# PR: Support-Ticket Blog Large Upload Policy Validation

## Why this slice exists

PR #990 proved that small no-outcome/no-resolution support-ticket uploads can
use a compact blog quality policy instead of forcing a 1500-word article. The
same change must not quietly shrink larger support-ticket uploads, because a
larger customer CSV can support a fuller descriptive article even when it lacks
resolution text or measured outcomes.

This slice pins the inverse contract at the source: compact policy applies only
to small support-ticket contexts, while larger or uncounted support-ticket
contexts keep the normal blog quality policy.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-provider-blog-polish
Slice phase: Functional validation

1. Add focused tests proving large no-outcome/no-resolution support-ticket
   contexts keep the base blog quality policy.
2. Add focused tests proving support-ticket contexts without a positive row
   count do not receive the compact small-upload exception.
3. Leave generation prompts and runtime implementation unchanged unless the new
   tests expose a source-level defect.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Support-Ticket-Blog-Large-Upload-Policy-Validation.md` | Plan doc for the inverse compact-policy validation slice. |
| `tests/test_extracted_blog_generation.py` | Pin large and uncounted support-ticket quality-policy behavior. |

## Mechanism

Exercise `_quality_policy_for_context` directly with support-ticket data
contexts:

- a no-outcome/no-resolution upload with `source_row_count` and
  `included_ticket_row_count` above the small-upload cap
- a no-outcome/no-resolution upload with no positive row-count signal

Both cases should return the original `QualityPolicy` object unchanged. This
keeps the guard close to the policy selector rather than relying on a live LLM
run to infer whether the threshold was applied.

## Intentional

- This is not another live Haiku validation. #990 already proved the end-to-end
  compact path. This slice protects the complementary branch with deterministic
  tests so future prompt/policy work cannot collapse large uploads into the
  compact path.
- This does not change the current `any(count <= cap)` behavior for mixed row
  counts. Packaged support-ticket inputs can intentionally include a small
  included subset from a larger source, and that remains outside this narrow
  inverse check.
- No `ATLAS-HARDENING.md` item is added because this slice closes a test gap
  without discovering a separate non-blocking risk.

## Deferred

- Broader acceptance testing across many customer CSV shapes remains a later
  robust-testing slice.
- Customer-language keyword promotion and standalone FAQ Article output remain
  future product slices coordinated with the FAQ lane.
- Parked hardening: none planned.

## Verification

- `python -m pytest tests/test_extracted_blog_generation.py -q`
  - `64 passed`
- Local PR review command with the temporary PR body file
  - passed

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~70 |
| Tests | ~45 |
| **Total** | **~115** |

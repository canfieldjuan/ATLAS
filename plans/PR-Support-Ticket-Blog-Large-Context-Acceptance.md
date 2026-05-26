# PR: Support-Ticket Blog Large Context Acceptance

## Why this slice exists

PR #991 pinned the compact-policy selector directly, but the production path is
route provider -> blog dispatcher -> blog generation service -> quality gate.
We still need a deterministic acceptance proof that a larger support-ticket
upload carries large row counts through the blog execute handoff and that the
blog generation service keeps the default 1500-word floor for that context.

This is the next robust-testing slice after the small-upload live validation:
prove the large-upload branch remains large all the way to the quality gate,
without spending another live LLM call.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-provider-blog-polish
Slice phase: Robust testing

1. Extend the loader-backed blog execute stress test so the captured blog
   service call asserts large support-ticket row counts in the blog data context.
2. Add a deterministic blog-generation service test proving a large
   no-outcome/no-resolution support-ticket context still blocks short content
   with the default 1500-word floor.
3. Keep runtime code unchanged unless those acceptance tests expose a real
   wiring defect.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Support-Ticket-Blog-Large-Context-Acceptance.md` | Plan doc for the large support-ticket blog context acceptance slice. |
| `tests/test_support_ticket_provider_landing_blog_execute.py` | Assert large loader-backed blog execute calls pass bounded large row counts into the blog data context. |
| `tests/test_extracted_blog_generation.py` | Prove large support-ticket blog generation keeps the default 1500-word quality floor. |

## Mechanism

The route-level test reuses the existing 50,000-row loader-backed execute
coverage. It already proves provider diagnostics are bounded; this slice adds
assertions on the captured blog service kwargs so the large counts are shown to
survive the handoff into the blog data context.

The service-level test passes a large trusted support-ticket data context into
BlogPostGenerationService.generate and feeds a deterministic 1,000-ish word
draft. If the small-upload compact policy accidentally applies, the draft would
not produce the content-too-short blocker requiring 1500 words. The test asserts the default
1500-word blocker, which proves the large branch reached the quality gate.

## Intentional

- No live Haiku run in this slice. The small-upload path already has live proof;
  this slice tests large-upload wiring and quality-policy selection
  deterministically.
- No FAQ generator or standalone FAQ Article changes. FAQ output shape remains
  owned by the parallel FAQ session.
- No runtime code change is expected. If the tests fail, the fix will happen at
  the source of the wiring/policy defect rather than by weakening assertions.

## Deferred

- Broader customer CSV acceptance testing across varied real exports remains a
  later robust-testing slice.
- Customer-language keyword promotion and standalone FAQ Article output remain
  future product slices coordinated with the FAQ lane.
- Parked hardening: none planned.

## Verification

- `python -m pytest tests/test_support_ticket_provider_landing_blog_execute.py tests/test_extracted_blog_generation.py -q`
  - `76 passed`
- Local PR review with PR body
  - passed

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~75 |
| Route acceptance assertions | ~15 |
| Blog-generation acceptance test | ~65 |
| **Total** | **~155** |

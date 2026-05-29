# PR: Support Ticket Smoke Closeout

## Why this slice exists

The support-ticket input-provider lane just shipped the synthetic export chain:
header aliases, platform-shaped CSV fixture, operator docs, and count-only
contact-email diagnostics. The active backlog still reads like the next source
work might be another generic export slice, even though the remaining useful
proof is now blocked on real anonymized customer exports.

This slice updates coordination docs so future sessions do not keep polishing
synthetic smoke surfaces.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-input-provider
Slice phase: Product polish

1. Mark the synthetic support-ticket platform smoke chain as closed in the
   active deferred backlog.
2. Refresh the source-adapter audit recommendation to point away from further
   synthetic source-shape work.
3. Update per-product state so the next milestone reflects the real export
   dependency.

### Files touched

- `plans/PR-Support-Ticket-Smoke-Closeout.md` - Plan doc for this slice.
- `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md` - Record the support-ticket smoke chain closeout.
- `docs/audits/content_ops_source_adapter_audit_2026-05-16.md` - Refresh the source-adapter next-work recommendation.
- `docs/extraction/coordination/state.md` - Update Content Ops state and next milestone.

## Mechanism

The docs now state that PRs #1101, #1102, #1103, and #1105 closed the synthetic
support-ticket export coverage path. The remaining source-breadth task is
explicitly waiting for anonymized real customer exports, not more plausible
platform shapes.

## Intentional

- Docs-only closeout. No runtime, prompt, FAQ generation, or smoke behavior
  changes.
- This does not remove the real-export follow-up. It clarifies that the
  follow-up needs real samples before another implementation slice makes sense.
- This does not touch FAQ-owned work.

## Deferred

- Future PR: run the platform smoke against anonymized real customer exports
  when samples are available.
- Parked hardening: none.

## Verification

- Grep for closeout markers and stale next-step wording across the edited docs -
  passed; the docs now reference PRs #1101/#1102/#1103/#1105 and no longer keep
  the old "Next Content Ops work should come from a real host export fixture"
  wording in coordination state.
- Whitespace check - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~65 |
| Deferred backlog | ~15 |
| Source adapter audit | ~10 |
| Coordination state | ~10 |
| **Total** | **~100** |

This stays below the 400 LOC soft cap.

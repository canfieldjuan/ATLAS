# PR: Content Ops Source Adapter Cumulative Audit

## Why this slice exists

The source-adapter stream has added reviews, calls, CRM rows, renewals,
tickets, surveys, bundles, nested threads, and tolerant field aliases. The
individual PRs were small, but the cumulative adapter now has enough branches
and key lists that another shape PR risks completionism instead of product
leverage.

## Scope

1. Add a current audit of the source-adapter shape, supported surfaces, risks,
   and refactor triggers.
2. Update the active AI Content Ops backlog so the next source work requires
   demand signal or a small refactor decision.
3. Keep the change docs-only.

## Mechanism

The audit records the current branch/key counts, what downstream paths consume
the normalized opportunities, what is still intentionally generic, and the
criteria for choosing more source breadth versus adapter consolidation.

## Intentional

- No source-adapter code changes.
- No generated-asset, API, database, or frontend changes.
- No attempt to refactor the adapter in this slice.

## Deferred

- A data-driven source-type registry.
- End-to-end generation tests that compare output quality by source type.
- Provider-specific importers for named CRM/support/call platforms.

## Verification

- Run the local PR review script.
- Run diff whitespace checks.

### Files Touched

- `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md`
- `docs/audits/content_ops_source_adapter_audit_2026-05-16.md`
- `plans/PR-Content-Ops-Source-Adapter-Audit.md`

## Estimated Diff Size

| Area | Estimated LOC |
|---|---:|
| Audit doc | ~120 |
| Backlog and coordination | ~25 |
| Plan doc | ~55 |
| **Total** | ~200 |

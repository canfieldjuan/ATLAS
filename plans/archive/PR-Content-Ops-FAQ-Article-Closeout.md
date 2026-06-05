# Content Ops FAQ Article Closeout

## Why this slice exists

PR #673 shipped richer support-ticket FAQ Markdown articles, but the
coordination ledger still shows `PR-Content-Ops-FAQ-Article-Renderer` as
in-flight and the deferred backlog still stops at the older FAQ output-check
closeout. That stale state can cause another session to avoid already-merged
FAQ files or reopen work that is complete.

## Scope (this PR)

1. Remove the merged FAQ article-renderer row from the in-flight coordination
   table.
2. Record the richer FAQ article renderer closeout in the AI Content Ops
   deferred backlog.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Article-Closeout.md` | Plan doc for this slice. |
| `docs/extraction/coordination/inflight.md` | Remove stale merged FAQ article-renderer row. |
| `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md` | Add PR #673 FAQ article-renderer closeout note. |

## Mechanism

This is docs-only state reconciliation. The in-flight table returns to an empty
coordination state, and the backlog's FAQ output-contract paragraph now names
the PR #673 behavior: summaries, numbered next steps, support escalation
guidance, and cited ticket quotes.

## Intentional

- No production code changes. The runtime behavior already shipped in PR #673.
- No new tests. This slice only updates coordination and audit text.

## Deferred

- None. Future FAQ/source work should still wait for a real customer help desk
  export or hosted UI need, as the backlog already states.

## Verification

- `bash scripts/local_pr_review.sh --allow-dirty` -> passed

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Content-Ops-FAQ-Article-Closeout.md` | 50 |
| `docs/extraction/coordination/inflight.md` | 4 |
| `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md` | 7 |
| **Total** | **61** |

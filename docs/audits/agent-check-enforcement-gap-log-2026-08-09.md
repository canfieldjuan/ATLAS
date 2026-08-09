# Agent Check Enforcement Gap Log - 2026-08-09

## Purpose

Log why the existing drift, compaction, PR ownership, and hardening/polish
deferral checks are not consistently respected. Ground truth here is the
current repository code and workflow docs, not the intended behavior.

## Current verdict

The rules exist, but several are still mostly reactive. They catch malformed PR
artifacts, mismatched lanes, missing reconciliation records, and oversized file
sets after an agent has already acted. The repeated failures come from three
gaps:

1. Local session state is mandatory by policy, but invisible to CI.
2. PR mutation ownership is enforced in some wrappers, but not all mutation
   paths.
3. Hardening/polish deferral is a judgment policy with fixture tests, but no
   live adapter that forces the correct "waive/defer" decision for real review
   comments.

## Enforcement Map

| Area | Rule or check that exists | What it prevents today | Why it is not being respected | Better preventative hook |
|---|---|---|---|---|
| Review boundary | `AGENTS.md:42` says stop at the Review Contract boundary; `AGENTS.md:64` says waive out-of-scope hardening, duplicates, speculative risks, and style-only issues. | Gives the reviewer and builder a written rule for when not to expand a PR. | It is prose plus synthetic policy. The Codex connector can still file the thread, and the builder still sees `live-reconciliation` red until the thread is fixed or waived. Nothing forces "waive this" before code is changed. | Add a review-disposition preflight for bot threads that classifies each thread as fix vs waive before editing, and require that decision record before any fix-loop push. |
| Duplicate/instance sprawl | `AGENTS.md:54` says report the class, not the instance, and merge "fresh evidence" into the existing finding. `AGENTS.md:1337` allows at most one finding per root defect class. | Establishes a no-thread-sprawl rule for reviewer behavior. | The current live gate checks whether threads are reconciled, not whether the reviewer created several threads for the same root decision. The builder still has to disposition every thread. | Add live reconciliation grouping: fail or warn when multiple open bot threads map to the same root decision without `waived-duplicate` entries. |
| Compaction drift | `AGENTS.md:431` makes one local session-state file mandatory; `AGENTS.md:441` requires updates after compaction/restart; `docs/SESSION_STATE_TEMPLATE.md:86` contains a resume checklist. | Gives a compacted/restarted agent a baton and tells it what to read before PR actions. | The state file is gitignored local state (`docs/SESSION_STATE_TEMPLATE.md:8`), so GitHub Actions cannot see it. No wrapper currently proves the checklist was read or updated before comment inspection, push, or merge. | Add a local `session_preflight` wrapper that requires `ATLAS_SESSION_STATE_FILE`, checks freshness/owned PR/current lane, and make push/comment-resolution/merge helpers call it. |
| PR lane ownership | `AGENTS.md:454` requires open-PR, main-log, owned-PR, branch/head checks before PR mutation. `scripts/check_session_pr_ownership.py:84` rejects PRs not listed under owned/may-touch state. | Prevents a session from mutating a PR that is not in its local ownership map when the guard is called. | The guard is only useful at the mutation boundary. `scripts/open_pr.sh:484` calls it for existing PR body edits, but `scripts/push_pr.sh:96` runs branch/body/local review without checking target PR ownership. Raw `git push`, `gh pr edit`, thread resolution, and `gh pr merge` bypass it unless the agent remembers the rule. | Move ownership checking into a shared mutation guard and call it from push, open/edit, merge, and thread-resolution helpers. Reject when `ATLAS_SESSION_STATE_FILE` is missing for PR work. |
| Lane/phase drift | `scripts/local_pr_review.sh:299` runs `audit_pr_session_drift.py` with `--require-current-pr-body`. The drift audit fails on ownership/phase/path overlap errors (`scripts/audit_pr_session_drift.py:112`). | Catches branch PR-body mismatches, open-PR file overlaps, and lane overlaps that are visible from plan docs and GitHub PR metadata. | It only sees committed plan docs and PR bodies. It does not read the local session-state file, cannot know that a compacted agent skipped the baton, and extracts branch phases only from changed plan docs (`scripts/audit_pr_session_drift.py:266`). | Keep this gate, but do not treat it as the compaction/ownership guard. Pair it with the local mutation preflight above. |
| Slice phase semantics | `AGENTS.md:713` defines phases and says workflow/process/hardening must justify why that phase is appropriate now. `scripts/audit_pr_session_drift.py:403` validates phase names. | Prevents invalid phase labels. | The audit checks allowed strings, not whether "Production hardening" is justified by a real blocker/risk/failed run, or whether a "Vertical slice" is carrying hardening/polish work. | Add a plan-body semantic audit for hardening/process phases: require a named blocker/risk/failed run, and fail when Deferred/Parked hardening is empty after a review-thread fix loop that touched non-root files. |
| Fix-loop growth | `AGENTS.md:1288` requires a fix baton with allowed files and a max-files budget. `scripts/audit_plan_doc_files_touched.py:90` can read `Max files: N` and fails when actual files exceed it (`scripts/audit_plan_doc_files_touched.py:160`). | Can cap changed file count when `Max files: N` is declared in the plan Scope. | The budget is optional. If the plan does not declare `Max files: N`, the audit returns no budget (`scripts/audit_plan_doc_files_touched.py:98`) and cannot stop the PR from growing. The real fix baton lives in local ignored session state, not CI. | Require `Max files: N` during PR fix mode, or whenever AI reconciliation is non-empty after the initial PR open. |
| Hardening/polish deferral | `AGENTS.md:746` allows inline fixes only for flow breakage, contract/CI violations, security, misleading output, or reviewer BLOCKERs; `AGENTS.md:757` sends everything else to `HARDENING.md`. | Provides a clear triage rule for the builder. | The plan/body audits enforce section shape, not whether the agent correctly parked non-blocking hardening. The easiest path under red AI reconciliation is still to code until the thread disappears. | Add a required "Disposition before edit" block for review comments: root cause, blocker predicate, fix/waive decision, allowed files, and parking target. |
| Codex scope policy fixtures | `scripts/codex_review_scope_policy.py:61` includes duplicate, out-of-scope hardening, speculative, nit, adjacent-edge, and true-blocker fixtures. `AGENTS.md:1359` says these are deterministic fixture oracles, not adapters. | Proves the desired policy in synthetic cases. | The fixtures are not wired to live comments. They cannot stop Codex from filing a broad comment, and they cannot stop a builder from treating a valid waiver as required code. | Build the missing adapter layer: feed live thread summaries into the same disposition vocabulary and require the chosen disposition before code changes. |
| AI reconciliation | `AGENTS.md:1378` makes `live-reconciliation` the gate for unresolved Codex threads. `scripts/audit_ai_reconciliation.py:68` allows `fixed-in`, `waived-duplicate`, `waived-out-of-scope`, `waived-speculative`, `waived-nit`, and `not-applicable`. | Prevents merging with unaccounted bot threads. | It is intentionally reactive: it proves every thread has an accounting record. It does not decide whether the correct record is `fixed-in` or `waived-*`, so agents can over-fix. | Keep the gate, but add a separate reviewer-scope/disposition gate before edits. |

## Why This Keeps Showing Up In PRs

The machine-enforced checks are better at artifact correctness than at agent
behavior. They can prove "the PR body has a reconciliation section" or "the
plan's file list matches the diff." They cannot currently prove "the agent read
the compaction baton," "the agent owns this PR before resolving a thread," or
"this comment was hardening and should have been waived instead of fixed."

That creates a bad incentive loop:

1. Codex files a broad or duplicate thread despite the written boundary.
2. `live-reconciliation` goes red because a thread exists.
3. The builder treats red as "must edit" instead of first classifying the thread.
4. The PR grows, which gives Codex more changed lines and more chances to file
   another edge/hardening comment.

The rules are present; the missing piece is an enforced decision checkpoint
before edits in a fix loop.

## Preventative vs Reactive Split

Preventative today:

- Branch/body admission in `push_pr.sh` and `open_pr.sh`.
- Local review before push.
- Plan file/claimed-files/diff-budget checks when the plan declares a budget.
- Existing PR body edit ownership when `open_pr.sh` updates an existing PR.

Reactive today:

- `session-lane` / drift checks after branch or PR artifacts exist.
- `live-reconciliation` after Codex has already left comments.
- AI reconciliation body audit after the builder has chosen fix vs waive.
- Hardening/polish parking rules after the agent has already noticed adjacent
  work.

Not meaningfully enforced yet:

- Compaction resume checklist completion.
- Session-state freshness before PR mutation.
- PR ownership before push, merge, or thread-resolution commands.
- Required `Max files: N` in fix loops.
- "Waive/defer before editing" for hardening, polish, duplicate, speculative, or
  adjacent-edge comments.

## Recommended Next Slice

Build a small workflow/process slice named `fix-loop-disposition-preflight`.
Keep it under the workflow lane and make it preventative:

1. Add a shared local mutation preflight script that requires
   `ATLAS_SESSION_STATE_FILE`, parses owned PR/lane/head, and can be called by
   push, merge, and comment-resolution helpers.
2. Add a fix-loop disposition file or PR-body block that must exist before a
   push made after bot comments or red AI reconciliation.
3. Require `Max files: N` when that fix-loop disposition exists.
4. Add synthetic fixtures for the missing behavior: duplicate thread,
   out-of-scope hardening, adjacent edge after contract met, and valid BLOCKER.
5. Wire only the smallest wrapper path first, then use the fixtures to prove the
   rule blocks "edit first, classify later."

This should reduce thread-chasing because the builder has to make the
fix-vs-waive call before changing code, and the wrapper can reject scope growth
before GitHub gets another larger diff to review.

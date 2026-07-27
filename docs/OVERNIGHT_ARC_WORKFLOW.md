# Overnight Arc Workflow

Turn "task + overnight" into an unattended arc that ends **merged** or
**cleanly blocked with a morning report**. Task-agnostic: a contract-shaped
GitHub issue, a feature, a refactor, or a read-only investigation. This doc is
canonical and survives sessions/compaction because it lives in the repo and is
wired into `AGENTS.md` (section 3c.2) and the `CLAUDE.md` compact-instructions
baton. Machine-local session tooling (kickoff prompts, local watcher copies)
may mirror it but this file wins on conflict.

The one rule that makes overnight work: **every question gets asked at
pre-flight, while the operator is still awake. After kickoff, zero operator
contact until the morning report** (except the true-blocker channel, section 3).

## 1. Pre-flight (with the operator, ~10 minutes, BEFORE the night starts)

Run this checklist interactively. Do not start the night on a task that fails
it.

1. **Task named.** An issue number, or a stated goal. If it is a goal, not an
   issue: write the contract down NOW (step 2) and get a yes on it.
2. **Readiness check** -- the task has (or you now write, and the operator
   confirms):
   - Problem-derived contract: what is broken / wanted; root cause if known.
   - Correct-work-must: the observable outcomes that define done.
   - Must-not-change: product-shape freezes, do-not-touch files/lanes,
     protected behavior. Check the active-lanes state (session handoff docs /
     open PRs) for collisions and record them here.
   - Acceptance: how the morning verdict is proven (tests, probes, e2e).
   - Scope split: what is explicitly OUT (named follow-up, not silent drop).
   The exemplar shape is issue #2060 -- that structure is why its arc ran
   multiple review rounds unattended.
3. **Ask every clarifying question now.** Anything that would otherwise be a
   3am guess: error-direction choices, product-shape edges, budget (LOC / PR
   count), whether multi-PR is acceptable. Vague answers get tightened before
   kickoff, not during.
4. **Authorization recorded in the session, explicitly:** overnight arc
   assigned per the operator's standing autonomous-arc rules -- merge-on-green
   authority for THIS arc's PRs, hardened-path defaults, defer-do-not-ask.
   Plus any per-night limits (files not to touch, stop-after-N-PRs).
5. **Mechanics armed:**
   - Fresh session for the arc (one arc per session; long-lived sessions
     degrade -- model fallback, compaction damage).
   - `git fetch` + working-tree provenance check; build in an own worktree
     (`git worktree add worktrees/<slice> -b claude/pr-<slice> origin/main`);
     never build on the shared main checkout.
   - Owned-PR watcher available: `scripts/watch_owned_pr.sh` (see section 5).
   - **Wake path verified for the builder surface** (AGENTS 3c.1.2): a
     watcher process does not wake an agent by itself. Claude Code native
     mode wakes on background-task completion; Codex/local CLI mode needs an
     external wake bridge. If the surface has no working wake path, do NOT
     launch the overnight arc from it -- the night would stall at the first
     watcher exit.

## 2. The night loop

Run the standard builder contract (`AGENTS.md` section 3: plan doc first,
diff budgets, PR body shape, reconstruction-review coding) with these
overnight deltas:

- **Never wait on the operator.** Technical fork -> hardened path, note the
  decision in the PR. Genuinely-operator choice -> GitHub issue, keep moving
  on everything else. Chat is write-only until morning.
- **Slices sequential, one at a time.** Multi-slice arcs finish slice N
  (merged or parked-with-issue) before opening slice N+1. Keep each PR inside
  the diff budget (<400 LOC, ~40 changed Python symbols).
- **Watcher-driven waiting.** After every push: resolve verified-fixed review
  threads, then arm `scripts/watch_owned_pr.sh` on the new head and stop
  working until it exits. Re-arm after EVERY push and EVERY compaction. No
  inline polling or sleep loops (AGENTS 3c.1.3).
- **Review rounds:** reconcile every bot/reviewer finding by execution, not
  narrative. Hard 3-round bot-review cap counted by rounds: at cap, fix
  verified findings, waive the rest with reasons in the PR body (AGENTS
  4a.1), merge on required-green. EXCEPTION: confirmed fail-open findings in
  money/auth/PII guards block past the cap; the move there is a structural
  acceptance bar (a fail-closed invariant + a generative property test), not
  a fourth round of spot patches.
- **Guard-shaped slices** (validator/sanitizer/privacy/cap): self-apply the
  reviewer boundary probe BEFORE pushing -- both error directions (fail-open
  AND over-reject), boundary values, at least one negative test. Cheaper than
  a review round.
- **Merge:** required checks green + 0 unresolved Codex threads + no
  CHANGES_REQUESTED + clean tree + local==remote -> merge, teardown, next
  slice. Log for the report; do not ping the operator per merge. Exception:
  documentation-only PRs hold for bot review before merging (doc PRs that
  merged on green before the bot pass have burned us).
- **Compaction recovery:** the `CLAUDE.md` compact instructions preserve the
  overnight baton. First acts after any compaction: re-read this doc, verify
  the baton against `git`/`gh` reality, re-arm the watcher.
- **Read-only investigation arcs:** no mutations -- findings become
  contract-shaped GitHub issues (future overnight tasks) + the report.

## 3. True-blocker channel (the only overnight contact)

Only when the WHOLE arc cannot proceed (a single blocked slice gets parked as
an issue while other slices continue) AND the decision is genuinely the
operator's: open a `BLOCKED: <decision>` issue, assign + @mention the
operator, and send the direct email alert if configured. Then keep doing
whatever work remains.

## 4. Morning report (the arc's last act, always -- success or not)

```
OVERNIGHT ARC REPORT -- <task> -- <date>
Outcome: CLEARED | PARTIAL | BLOCKED
Merged:   #PR (title, LOC, merge SHA) ...
Open:     #PR @ state (what it awaits)
Issues:   filed #N (deferred decision / follow-up / BLOCKED)
Review:   rounds per PR; findings fixed/waived (waivers listed)
Verification: what was EXECUTED to prove acceptance (commands/tests)
Left for operator: decisions only they can make, one sentence each
Next:     recommended next arc or none
```

Post it in chat; email it too if the arc ended BLOCKED.

## 5. Owned-PR watcher

`scripts/watch_owned_pr.sh` -- portable single-PR watcher for the builder
side (companion to the reviewer lane's inbox watcher, which is session
tooling). Usage:

```bash
PR=<number> SHA=<full 40-char head sha> bash scripts/watch_owned_pr.sh
```

Run it in the background (no trailing `&` inside the command -- background it
at the harness level). It polls about every 29 minutes and exits the moment
there is something to act on:

- `MERGED/CLOSED` -- terminal; stop watching.
- `HEAD-MOVED` -- the branch advanced past the armed SHA; reconcile, re-arm
  on the new head.
- `ACTIONABLE` -- a red required context, unresolved review threads (counts
  fail closed when more thread pages exist than fetched), or a
  CHANGES_REQUESTED review decision.
  Definite negatives exit on any cycle, including the first.
- `MERGE-READY` -- readiness is presence-based and fail-closed: EVERY
  required branch-protection context (read at runtime from
  `scripts/check_required_status_checks.py`, so the gate cannot drift from
  the canonical list) must be present and reporting success, plus
  0 unresolved Codex threads, no CHANGES_REQUESTED, and mergeable. A context
  that has not started yet keeps readiness false. Run the pre-merge checklist
  (clean tree, local==remote, threads still 0), merge, alert.

The watcher itself never merges and never holds merge authority (AGENTS
3c.1.1); it only reports states.

## 6. Kickoff prompt (paste into a fresh session at night)

> Read docs/OVERNIGHT_ARC_WORKFLOW.md and run tonight's overnight arc:
> <task / issue #>. Pre-flight is done: <contract or issue #>. You have
> merge-on-green authority for this arc's PRs. Report at morning per
> section 4.

If pre-flight was not done yet, the fresh session runs section 1 with the
operator before they leave -- that is the intended flow.

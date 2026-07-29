# PR-Codex-Issue-Queue-Handoff

## Why this slice exists

#1999 shipped the Codex wake foundation but explicitly deferred "issue-queue
continuation plus defer/email notification for operator-owned decisions." The
operator then approved the plan to make long-horizon Codex sessions continue
from a GitHub issue queue and defer genuinely operator-owned decisions into a
trackable issue plus an email-ready local alert artifact.

Root cause: the wake bridge can resume a Codex session on PR state, but after a
merge or an operator-only fork there is no repo-owned next-work source or
machine-checkable defer artifact. That leaves continuation dependent on chat
memory or desktop notifications.

This fixes the root for the workflow/process lane by making GitHub Issues the
queue source of truth and by giving operator-owned defers a concrete artifact.
It does not change product behavior.

This exceeds the 400 LOC soft cap because the slice needs the CLI, negative
tests for the parser/transport safety boundary, CI enrollment, and docs in one
PR. Splitting those would create the same half-wired workflow tool this lane is
trying to prevent.

## Scope (this PR)

Ownership lane: workflow/codex-autonomy
Slice phase: Workflow/process

1. Add a `scripts/codex_issue_queue.py` CLI that selects the next eligible
   issue for a lane and records operator-owned defers.
2. Use GitHub Issues as the queue source: open issues with the `codex` label,
   an `Autonomy lane: <lane>` marker, optional `Autonomy priority: <int>`, and
   no `deferred` label.
3. Write local email-ready defer artifacts under the user state directory
   without sending email or requiring secrets.
4. Document the queue/defer handoff in the Codex watcher handoff and reusable
   playbook docs.
5. Add focused tests that prove ordering, fail-closed ambiguity, defer comments,
   local artifact output, and no merge/push command path.

### Review Contract

- [ ] The CLI can discover a single next queued issue from mocked GitHub issue
      data using trusted queue labels/comment associations, lane markers, and
      priority ordering.
- [ ] The CLI fails closed when there is no eligible issue or when issue data is
      ambiguous/malformed.
- [ ] The defer path writes a local email-ready artifact, persists a durable
      `deferred` label, and posts a quoted GitHub issue comment, but does not
      send email or mutate PR/branch state.
- [ ] The docs tell Codex/local sessions to use the issue queue after merge and
      to defer only operator-owned decisions.
- [ ] Triggered reviewer rules: R1, R2, R4/R5 for command safety, R9/R14.

### Files touched

- `.github/workflows/codex_wake_bridge_checks.yml`
- `docs/autonomous_coding_repo_playbook.md`
- `docs/long_running_session_watcher_handoff.md`
- `plans/PR-Codex-Issue-Queue-Handoff.md`
- `scripts/codex_issue_queue.py`
- `tests/test_codex_issue_queue.py`

## Mechanism

The CLI shells out to `gh issue list`, `gh issue edit`, and `gh issue comment`
only. `next` server-filters GitHub Issues by the `codex` queue label before the
client-side priority sort, then parses trusted issue body/comments for:

- `Autonomy lane: <lane>`
- `Autonomy priority: <int>` (optional)

`next` filters to open issues in the requested lane without the `deferred`
label or `Autonomy deferred: true` marker, sorts by priority then updated time,
and prints JSON plus an optional concise handoff summary. Issue body markers
count only on `codex`-labeled issues; comment markers count only from trusted
GitHub author associations (`OWNER`, `MEMBER`, `COLLABORATOR`). `defer` writes
the local Markdown artifact before GitHub mutation, adds the durable `deferred`
label, then posts a quoted issue comment so multiline freeform text cannot
inject queue-control markers. Network and GitHub transport are tested by
mocking `subprocess.run`; selection/defer logic stays real.

## Intentional

- No real email send in this slice. The operator selected "issue + artifact" as
  the safe first step; wiring Resend/Gmail would add credentials and a larger
  blast radius.
- No repo YAML queue. GitHub Issues plus the maintainer-applied `codex` queue
  label are already the source of truth for #1962 and survive across
  sessions/tools.
- No automatic PR creation or merge. The CLI only selects/records handoffs.

## Deferred

- Real email/ntfy/desktop alert delivery for operator-owned defers remains a
  follow-up after the issue/artifact contract has review mileage.
- Optional GitHub label automation for queued/autonomy issues remains deferred;
  the v1 contract expects maintainers to apply the `codex` queue label before
  body markers are honored.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_codex_issue_queue.py tests/test_codex_wake_bridge.py tests/test_install_codex_wake_bridge.py tests/test_audit_pr_watcher_safety.py -q` -- 55 passed.
- `python scripts/maturity_sweep.py scripts --tests-root tests --baseline tests/maturity_sweep/baseline_scripts.json --min-score 8 --sensitive-glob 'scripts/**'` -- ratchet gate passed.
- `ATLAS_SESSION_STATE_FILE=SESSION_STATE.codex-issue-queue-1962.local.md bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-body-codex-issue-queue-handoff.md` -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/codex_wake_bridge_checks.yml` | 3 |
| `docs/autonomous_coding_repo_playbook.md` | 13 |
| `docs/long_running_session_watcher_handoff.md` | 14 |
| `plans/PR-Codex-Issue-Queue-Handoff.md` | 120 |
| `scripts/codex_issue_queue.py` | 279 |
| `tests/test_codex_issue_queue.py` | 278 |
| **Total** | **707** |

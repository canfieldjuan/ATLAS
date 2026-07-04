# PR-Watcher-Safety-Handoff

## Why this slice exists

The long-running watcher setup stopped making progress overnight because the
local watcher/desktop-notification path was treated like a wake-up mechanism,
but it did not create a durable handoff an active agent could consume. At the
same time, one stale local watcher path still carried merge behavior even after
the repo docs said the watcher must not auto-merge.

Root cause: the no-merge rule lived mostly in prose and in the operator's
current machine state, not in repo-enforced tooling. A local watcher executable
or config could drift back into merge authority, Claude Code and Codex/local
sessions were being described with the same wake rules, and an active builder
had no single read-only command that summarized "ready, attention, pending,
stale" watcher state on resume.

This fixes the root for this workflow layer by making watcher merge authority a
blocking local-review failure and by adding a read-only ready-state handoff
command for active builders. The watcher can only write state; the active
builder consumes that state, reruns AGENTS guards, and merges only when the
operator has explicitly authorized the active builder for that arc.

It also separates the two builder surfaces: Claude Code uses native PR
subscription/review reactivity plus 30-minute polling, while Codex/local CLI
sessions need an external wake bridge if they are expected to resume
autonomously from watcher state.

This is over the soft 400 LOC target because the workflow rule, the blocking
audit, the read-only handoff reporter, CI/local-review enrollment, and the
negative fixtures are one load-bearing unit. Splitting them would recreate the
same doc-only/process-only gap this slice is meant to close.

## Scope (this PR)

Ownership lane: workflow/autonomous-ci-cd-map
Slice phase: Workflow/process

1. Add a local-review audit that blocks watcher merge authority in repo docs,
   local watcher source, and local watcher configs.
2. Add a read-only watcher ready-state reporter that active builders run on
   resume before acting on watcher output.
3. Update long-running-session docs and the session-state template so the
   watcher writes status only and the active builder owns guarded merge
   decisions.
4. Separate Claude Code native subscription/polling from Codex/local
   watcher-state handoff so Claude sessions do not inherit local timer
   requirements.

### Review Contract

Acceptance criteria:

1. Repo docs/templates say local watchers are status-only and never carry merge
   authority.
2. Local review runs a watcher-safety audit that fails on truthy
   `AUTO_MERGE`, watcher/wrapper/drop-in merge/delete-branch commands, or docs
   that give any wake surface merge authority.
3. The audit has negative fixtures proving those failure branches fire, plus a
   repo-only fixture so CI/local review still passes when no local watcher is
   installed.
4. The active-agent handoff command reads watcher JSON and buckets ready,
   attention, pending, stale, and other states without mutating GitHub or local
   watcher files.
5. The handoff reporter has fixtures for ready, attention/failure,
   contradictory ready snapshots, bridge handoff artifacts, stale merged PRs,
   and archived state directories.
6. Docs state that true Codex/local autonomy requires an external wake bridge
   that starts or resumes Codex with watcher state; `atlas-pr-watch` alone is
   state-only.

Affected surfaces:

- Long-running agent docs and session-state template.
- Local pre-push/local-review audit bundle.
- Local watcher state inspection scripts.

Risk areas:

- False-positive doc matching that blocks safe text.
- Accidentally treating watcher readiness as merge authority.
- Accidentally forcing Claude Code sessions into the Codex/local timer path.
- Local-review drift if the audit is not enrolled.

Reviewer rules triggered: R1, R2, R10, R13, R14.

### Files touched

- `.github/workflows/pre_push_audit.yml`
- `AGENTS.md`
- `docs/SESSION_STATE_TEMPLATE.md`
- `docs/long_running_session_watcher_handoff.md`
- `plans/PR-Watcher-Safety-Handoff.md`
- `scripts/audit_pr_watcher_safety.py`
- `scripts/pre_push_audit.sh`
- `scripts/report_pr_watcher_state.py`
- `tests/test_audit_pr_watcher_safety.py`
- `tests/test_report_pr_watcher_state.py`

## Mechanism

`scripts/audit_pr_watcher_safety.py` scans the repo's watcher/session docs plus
the local watcher executable, wake wrapper, config files, and systemd unit/drop-in
files. It fails closed on three classes: truthy `AUTO_MERGE`, watcher source
containing PR merge/delete-branch commands, and repo docs/templates that grant a
watcher, timer, notification, or bridge merge authority.
`scripts/pre_push_audit.sh` runs this audit as part of the standard local-review
bundle, so the rule is enforced before every PR push.

`scripts/report_pr_watcher_state.py` reads local watcher JSON snapshots and
renders a handoff grouped by active-builder decision buckets: ready, attention,
pending, stale/closed, and other. It ignores bridge wake-handoff JSON artifacts
and classifies failure/pending details before accepting a ready snapshot,
matching the bridge's safety ordering. It is intentionally read-only. With
GitHub lookups enabled it refreshes PR closed/merged state; tests use
`--skip-github` to keep the behavior deterministic.

`.github/workflows/pre_push_audit.yml` explicitly enrolls the watcher safety
and handoff reporter tests in the PR-tooling unit-test list for both PR and main
jobs.

The docs now name the boundary: watcher processes may poll and write JSON/log
state, but only an active builder may act on that state after rerunning AGENTS
ownership/head/check/thread/reconciliation/mergeability guards. Claude Code
native sessions record Claude's subscription and 30-minute polling instead of
installing a local systemd timer by default. Codex/local sessions record whether
an external wake bridge exists; without that bridge, the watcher state is only a
handoff for the next active agent.

## Intentional

- The audit scans known repo docs instead of the entire repository. This keeps
  the safety check focused on workflow authority text and avoids unrelated
  markdown false positives.
- The reporter does not merge, resolve, edit, or archive anything. Stale state
  cleanup remains an active-builder action after inspection.
- The local watcher executable/config scan is skipped by `--repo-only` for
  deterministic tests and environments where a watcher is not installed.
- The Codex/local wake bridge now exists on `origin/main` via #1976. This slice
  treats it as a downstream consumer of watcher state and keeps this PR focused
  on the safety audit plus active-agent handoff reporter.

## Deferred

- Event-driven GitHub review webhooks are still outside this repo.
- Further bridge integrations, such as service-level event hooks that call
  `scripts/codex_wake_bridge.py --source event`, remain outside this slice.
  This PR only enforces the no-merge watcher boundary and the read-only
  ready-state handoff.

Parked hardening: none.

## Verification

- Pass: 29 tests passed:
  `python -m pytest tests/test_audit_pr_watcher_safety.py tests/test_report_pr_watcher_state.py -q`
- Pass: Python byte-compile for the changed watcher scripts and watcher tests.
- Pass:
  `python scripts/audit_pr_watcher_safety.py --repo-root . && python scripts/audit_pr_watcher_safety.py --repo-root . --repo-only`
- Pass: `python scripts/report_pr_watcher_state.py --skip-github`
- Pass:
  `python scripts/maturity_sweep_file_lane.py scripts/report_pr_watcher_state.py scripts/audit_pr_watcher_safety.py --tests-root tests --baseline tests/maturity_sweep/baseline_scripts.json --min-score 8 --sensitive-glob 'scripts/**'`
- Pass: local review bundle via `scripts/local_pr_review.sh`

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/pre_push_audit.yml` | 4 |
| `AGENTS.md` | 77 |
| `docs/SESSION_STATE_TEMPLATE.md` | 28 |
| `docs/long_running_session_watcher_handoff.md` | 174 |
| `plans/PR-Watcher-Safety-Handoff.md` | 179 |
| `scripts/audit_pr_watcher_safety.py` | 223 |
| `scripts/pre_push_audit.sh` | 1 |
| `scripts/report_pr_watcher_state.py` | 186 |
| `tests/test_audit_pr_watcher_safety.py` | 187 |
| `tests/test_report_pr_watcher_state.py` | 189 |
| **Total** | **1248** |

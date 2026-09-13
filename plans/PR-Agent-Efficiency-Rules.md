# PR-Agent-Efficiency-Rules

## Why this slice exists

The operator asked to stop spending model turns polling unchanged CI/review
state and to avoid duplicating broad CI test work locally. The first docs-only
implementation established that policy in `AGENTS.md`, but exact-head review of
PR #2521 exposed three reachable splits: mandatory companion instructions still
tell Claude to poll, the rewritten merge step no longer preserves the recorded
wake-source authorization boundary, and watcher readiness treats exact-head
Codex review evidence as diagnostic even though the new policy makes its absence
pending.

These are policy and merge-readiness failures in the current slice, not optional
hardening. Fixing only the cited sentences would leave the executable watcher
able to report a pre-review PR as ready.

The 400-LOC soft cap is exceeded because the root decision spans four mandatory
instruction surfaces, both watcher producers, the shared readiness consumer,
and their existing focused contract tests; splitting those pieces would publish
a policy/runtime contradiction or an unverified merge gate.

### Problem-derived contract

- Root cause: polling and review-readiness have multiple authorities. The first
  change updated `AGENTS.md` but left mandatory handoff docs on the former
  polling model. It also removed the activation-source predicate from the merge
  instruction. Both watcher producers already collect exact-head Codex review
  evidence, but their readiness decisions and the shared proof validator do not
  require complete evidence or a positive current-head attestation.
- Correct fix must touch/change: make every mandatory session instruction use
  subscription/external wake plus one snapshot per model activation; require the
  activation source to satisfy the recorded merge authorization; make the shell
  watcher, Python watcher, and shared readiness validator keep incomplete,
  absent, stale, or rejected exact-head review evidence out of ready state; and
  prove the positive and negative sides in focused watcher, bridge, and reporter
  tests.
- Must not change: watcher infrastructure remains read-only and never gains
  merge authority; external non-model timers may still collect state; required
  CI, thread pagination, ownership, mergeability, and reconciliation gates stay
  intact; no Atlas product/runtime/UI behavior or unrelated lane changes.

## Scope (this PR)

Ownership lane: atlas-agent-efficiency-policy
Slice phase: workflow/process

Max files: 12

1. Replace model-driven polling instructions with one-snapshot activation and
   preserve the scheduled-ready-only authorization boundary.
2. Make exact-head Codex review evidence a fail-closed readiness requirement in
   both watcher producers and the shared consumer predicate.
3. Add focused positive/negative tests for no review, stale/wrong review,
   incomplete review pagination, requested changes, and one valid exact-head
   review.

### Review Contract

- Acceptance criteria:
  1. `AGENTS.md`, `CLAUDE.md`, `docs/SESSION_STATE_TEMPLATE.md`, and
     `docs/long_running_session_watcher_handoff.md` consistently say that an
     active model takes one exact-head snapshot per activation and never acts as
     the timer; settled by a targeted `rg` audit.
  2. `AGENTS.md` permits merge only when the current activation source satisfies
     the authorization recorded in session state; an immediate or event wake
     cannot consume scheduled-ready-only authority.
  3. `tests/test_watch_owned_pr.py` proves the shell watcher emits
     `MERGE-READY` with one complete current-head Codex attestation and does not
     emit it when the attestation is absent, from the wrong identity/head, or
     incomplete.
  4. `tests/test_pr_watcher.py` proves the Python producer returns pending when
     no current-head attestation exists and attention when review collection is
     incomplete or changes are requested; one complete current-head review can
     still reach ready.
  5. `tests/test_codex_wake_bridge.py` and
     `tests/test_report_pr_watcher_state.py` prove version-1 readiness with
     incomplete review pagination or fewer than one exact-head review is not
     classified ready.
  6. `python scripts/audit_pr_watcher_safety.py --repo-only` proves no watcher,
     timer, bridge, or handoff gained merge authority.
- Reachability proof: the real shell/Python watcher entrypoints consume mocked
  GitHub snapshots in their existing focused test suites; observable output is
  `MERGE-READY` versus pending/actionable and ready versus pending/attention
  state/report buckets.
- Affected surfaces: builder instructions, session-state handoff, shell watcher,
  Python watcher, wake-bridge readiness validation, reporter classification, and
  their focused tests.
- Risk areas: premature merge readiness, stale/incomplete review evidence,
  mismatched wake authorization, accidental model polling, and watcher merge
  authority.
- Reviewer rules triggered: R1, R2, R5, R6, R8, R10, R12, R13, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: GitHub review evidence -> watcher state/output -> shared
  readiness validator -> reporter/wake classification -> active-builder merge
  decision.
- Replaced-path behaviors: zero or incomplete exact-head review evidence changes
  from ready/diagnostic to pending or attention; valid complete evidence remains
  ready; scheduled-ready-only authorization remains unavailable to other wakes.
- Guard-relevant fields: `codex_reviews_complete`,
  `codex_review_pages_fetched`, `codex_head_review_count`, `reviewDecision`, and
  activation source.
- Caller x input shape: shell watcher GraphQL pages, Python watcher version-1
  readiness dict, wake bridge, reporter, and active-builder instruction flow.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: no deployment/config fallback changes.
- Explicit value probe: complete current-head review evidence reaches ready.
- Absent value probe: missing/zero attestation remains pending or blocked.
- Default-session/default-context probe: an immediate post-push snapshot without
  a review cannot merge and cannot consume scheduled-only authorization.
- Side-effect ordering: all admission decisions remain before any active-builder
  merge; watchers remain status-only.

### Closure Declaration

- Readiness proof fields are **CLOSED** and **ENUMERATED** by the version-1
  readiness object constructed in `scripts/pr_watcher.py` and validated at the
  single choke point `scripts/codex_wake_bridge.py:readiness_blockers`. Missing,
  malformed, incomplete, or unrecognized proof data defaults to not-ready, the
  safe side of the merge decision.
- Wake sources are **CLOSED** and **DERIVED** from
  `scripts/codex_wake_bridge.py:classify_wake` (`event` versus `scheduled`). An
  unrecognized or authorization-mismatched source defaults to attention/no
  merge, the safe side.

### Files touched

- `AGENTS.md`
- `CLAUDE.md`
- `docs/SESSION_STATE_TEMPLATE.md`
- `docs/long_running_session_watcher_handoff.md`
- `plans/PR-Agent-Efficiency-Rules.md`
- `scripts/codex_wake_bridge.py`
- `scripts/pr_watcher.py`
- `scripts/watch_owned_pr.sh`
- `tests/test_codex_wake_bridge.py`
- `tests/test_pr_watcher.py`
- `tests/test_report_pr_watcher_state.py`
- `tests/test_watch_owned_pr.py`

## Mechanism

The Python producer classifies a clean PR with no exact-head Codex review as
pending, and classifies incomplete review evidence or requested changes as
attention. The shell watcher applies the same gates before and during its final
read. The shared version-1 proof validator independently requires complete
review pagination, at least one fetched review page, and at least one exact-head
Codex attestation, so legacy or contradictory ready snapshots fail closed in
both the wake bridge and reporter.

The instruction set distinguishes active model turns from external state
collectors: external timers/webhooks may wake a session, but the session takes
one snapshot and yields again on unchanged/pending state. Merge authorization
continues to be conditioned on its recorded activation source.

## Intentional

- A missing Codex review is pending, not actionable: there is nothing for the
  builder to fix until review evidence exists.
- Incomplete review API data or an open changes-requested decision is attention,
  because treating uncertainty as ready would weaken the merge gate.
- Local non-model watchers may continue polling GitHub; this slice removes model
  polling, not deterministic state collection.

## Deferred

- None.

Parked hardening: none.

## Verification

- Fail-first: four targeted regressions failed in the declared readiness class
  before implementation (`4 failed`).
- `uv run pytest -q tests/test_watch_owned_pr.py` - `26 passed`.
- `uv run pytest -q tests/test_pr_watcher.py` - `76 passed`.
- `uv run pytest -q tests/test_codex_wake_bridge.py
  tests/test_report_pr_watcher_state.py` - `68 passed`.
- `uv run ruff check scripts/codex_wake_bridge.py scripts/pr_watcher.py
  tests/test_codex_wake_bridge.py tests/test_pr_watcher.py
  tests/test_report_pr_watcher_state.py tests/test_watch_owned_pr.py` - all
  checks passed.
- `uv run ruff format --check ...` - not applied: it would bulk-reformat all
  six existing touched Python files, including unrelated baseline formatting.
- `bash -n scripts/watch_owned_pr.sh` - exit 0.
- `python scripts/audit_pr_watcher_safety.py --repo-only` - OK; watcher
  docs/config/source grant no merge authority.
- `python scripts/check_guard_class_closure.py --base origin/main --strict` -
  OK; no guard-shaped change without a property test.
- Targeted mandatory-doc audit - `stale model polling directives: 0`.
- `git diff --check` - exit 0.
- At push, `scripts/push_pr.sh` owns the single local review bundle; it is not
  duplicated manually.

## Estimated diff size

| File | LOC |
|---|---:|
| `AGENTS.md` | 86 |
| `CLAUDE.md` | 18 |
| `docs/SESSION_STATE_TEMPLATE.md` | 14 |
| `docs/long_running_session_watcher_handoff.md` | 34 |
| `plans/PR-Agent-Efficiency-Rules.md` | 222 |
| `scripts/codex_wake_bridge.py` | 11 |
| `scripts/pr_watcher.py` | 4 |
| `scripts/watch_owned_pr.sh` | 19 |
| `tests/test_codex_wake_bridge.py` | 20 |
| `tests/test_pr_watcher.py` | 75 |
| `tests/test_report_pr_watcher_state.py` | 18 |
| `tests/test_watch_owned_pr.py` | 64 |
| **Total** | **585** |

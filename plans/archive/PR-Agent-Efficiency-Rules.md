# PR-Agent-Efficiency-Rules

## Why this slice exists

The operator asked to stop spending model turns polling unchanged CI/review
state and to avoid duplicating broad CI test work locally. The first docs-only
implementation established that policy in `AGENTS.md`, but exact-head review of
PR #2521 exposed reachable splits: mandatory companion instructions still tell
Claude to poll, the rewritten merge step no longer preserves the recorded
wake-source authorization boundary, watcher readiness treats exact-head Codex
review evidence as diagnostic even though the new policy makes its absence
pending, and the canonical handoff example omits proof fields its consumer
requires. The shell watcher also reads final threads before final review
attestations, so a review created between those reads can contribute a positive
attestation while its new thread is absent from the readiness decision. Both
watcher classifiers let a missing review hide an already-known merge conflict.
A later exact-head review then exposed the same precedence split in the shell's
final-read path and showed that both producers discard an exact-head
`CHANGES_REQUESTED` review state when the aggregate PR decision is empty.
The next exact-head review showed that the new change-request signal is latched
across every historical review on the same commit instead of being reduced to
the connector's latest formal review state. It also showed that the boundary
declaration names caller groups without dispositioning each caller/input shape.
The corrected plan-required body then exposed a pre-push deadlock:
`--current-pr-body-file` validates the intended body but the session-drift audit
also validates the stale GitHub body for the same branch, while `open_pr.sh`
correctly refuses to update that body until the new head is pushed.
The latest exact-head review exposed two remaining contradictions in the same
merge gate: the final stability recheck performs review network reads after its
last unbound head read, and the mandatory overnight workflow exempts docs-only
PRs from the exact-head review presence the executable gate requires.
A subsequent exact-head review showed that the shell stability token covers
formal reviews but omits the clean top-level Codex comments the same gate also
accepts as attestations.

These are policy and merge-readiness failures in the current slice, not optional
hardening. Fixing only the cited sentences would leave the executable watcher
able to report a pre-review PR as ready.

The 400-LOC soft cap is exceeded because the root decision spans four mandatory
instruction surfaces, both watcher producers, the shared readiness consumer,
and their existing focused contract tests; splitting those pieces would publish
a policy/runtime contradiction, an unverified merge gate, or a body-update path
that cannot pass its own admission checks.

### Problem-derived contract

- Root cause: polling and review-readiness have multiple authorities. The first
  change updated `AGENTS.md` but left mandatory handoff docs on the former
  polling model. It also removed the activation-source predicate from the merge
  instruction. Both watcher producers already collect exact-head Codex review
  evidence, but their readiness decisions and the shared proof validator do not
  require complete evidence or a positive current-head attestation. The
  session-drift audit also treats its explicit local current-body override and
  the same branch's stale GitHub body as simultaneous authorities. Finally, the
  shell final-read order does not bind review evidence to a later thread
  snapshot, both classifiers prioritize pending review evidence over a definite
  negative merge state, the final shell read still prioritizes missing review
  over its other actionable facts, both review paginators collapse
  `CHANGES_REQUESTED` into the same zero-attestation result as no review, and
  the shell watcher checks the armed head before rather than after collecting
  its expanded final evidence snapshot. Both producers then validate only head
  identity, so a summary-only formal review can change on the same head after
  the accepted review snapshot without invalidating readiness. The stability
  rechecks themselves then fetch mutable review evidence after their last head
  observation without binding the returned evidence to that head, and the
  overnight workflow preserves a docs-only bypass of the exact-head gate. The
  shell then compares only formal reviews even though clean top-level Codex
  comments can independently satisfy the attestation requirement.
- Correct fix must touch/change: make every mandatory session instruction use
  subscription/external wake plus one snapshot per model activation; require the
  activation source to satisfy the recorded merge authorization; make the shell
  watcher, Python watcher, and shared readiness validator keep incomplete,
  absent, stale, or rejected exact-head review evidence out of ready state;
  document every required version-1 review-proof field; collect final review
  evidence before the final thread snapshot; surface definite merge conflicts
  before review-pending state; prove the positive and negative sides in focused
  watcher, bridge, and reporter tests; make final-read actionable facts precede
  review-pending; preserve exact-head change-request state through both
  producers and the versioned readiness proof; reduce ordered exact-head formal
  reviews to the connector's latest submitted state; enumerate every changed
  boundary caller/input disposition; bind the shell final-evidence snapshot to
  the armed head with a post-collection head check; and make the explicit local
  current-body override authoritative only for the current PR while preserving
  GitHub peer-overlap discovery; and require the complete exact-head formal
  review snapshot to remain stable across each producer's final evidence
  interval; require each paginated review/comment response to identify the
  armed head; remove the overnight docs-only review-presence exemption; and
  compare every accepted formal-review and clean-comment attestation source on
  both sides of the shell's final evidence interval.
- Must not change: watcher infrastructure remains read-only and never gains
  merge authority; external non-model timers may still collect state; required
  CI, thread pagination, ownership, mergeability, and reconciliation gates stay
  intact; no Atlas product/runtime/UI behavior or unrelated lane changes.

## Scope (this PR)

Ownership lane: atlas-agent-efficiency-policy
Slice phase: workflow/process

Max files: 16

1. Replace model-driven polling instructions with one-snapshot activation and
   preserve the scheduled-ready-only authorization boundary.
2. Make exact-head Codex review evidence a fail-closed readiness requirement in
   both watcher producers and the shared consumer predicate.
3. Add focused positive/negative tests for no review, stale/wrong review,
   incomplete review pagination, requested changes, and one valid exact-head
   review.
4. Remove the pre-push body-update deadlock by preventing a same-branch stale
   GitHub body from overriding the explicitly supplied local current-body file.
5. Make the handoff's canonical version-1 example match the enforced proof
   schema, bind final shell review evidence to a later thread snapshot, and
   surface definite merge conflicts before missing-review pending state.
6. Surface every final-read actionable fact before review-pending and propagate
   exact-head Codex `CHANGES_REQUESTED` states through both producers and the
   shared readiness proof.
7. Reduce exact-head formal reviews to the connector's latest submitted state
   and enumerate each changed readiness-boundary caller/input disposition.
8. Revalidate the armed head after the shell watcher collects every final
   review, thread, merge, and check fact so a mixed-head snapshot cannot report
   merge readiness.
9. Revalidate the complete exact-head formal-review snapshot after final
   evidence collection in both producers so a same-head review transition
   cannot inherit stale clean readiness.
10. Bind every review/comment page in both producers to the armed head so the
    final stability recheck cannot validate old-head evidence after a push.
11. Make the overnight direct caller require exact-head review evidence for
    documentation-only PRs just like every other owned PR.
12. Revalidate accepted clean top-level Codex comments alongside formal reviews
    before the shell watcher can report merge readiness.
13. Make the producer's actionable state outrank optional pending checks through
    one shared consumer classifier used by the wake bridge and state reporter.

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
  7. `tests/test_audit_pr_session_drift.py` proves an explicit valid local
     current-body file wins over the stale body of the same GitHub PR while the
     GitHub sweep still runs for peer PR conflicts.
  8. The canonical version-1 JSON example contains
     `codex_reviews_complete`, `codex_review_pages_fetched`, and
     `codex_head_review_count`; settled by comparison with
     `scripts/codex_wake_bridge.py:readiness_blockers`.
  9. `tests/test_watch_owned_pr.py` proves a Codex review submitted between the
     initial and final reads cannot contribute an attestation while its new
     unresolved thread is missed by the final readiness decision.
  10. `tests/test_watch_owned_pr.py` and `tests/test_pr_watcher.py` prove a
      definite dirty/conflicting merge state is actionable/attention even when
      no exact-head Codex review exists.
  11. `tests/test_watch_owned_pr.py` proves an unresolved final-read thread is
      actionable even when the final exact-head attestation disappears.
  12. `tests/test_watch_owned_pr.py`, `tests/test_pr_watcher.py`, and
      `tests/test_codex_wake_bridge.py` prove an exact-head Codex
      `CHANGES_REQUESTED` review is actionable and blocks readiness even when
      aggregate `reviewDecision` is empty.
  13. `tests/test_watch_owned_pr.py` and `tests/test_pr_watcher.py` prove the
      latest submitted exact-head formal Codex review controls the gate:
      requested-then-clean is accepted, while clean-then-requested is blocked.
  14. `tests/test_watch_owned_pr.py` proves a head change during final evidence
      collection emits `HEAD-MOVED` and never `MERGE-READY`; the Python producer
      remains covered by its existing post-review head-mismatch check.
  15. `tests/test_watch_owned_pr.py` and `tests/test_pr_watcher.py` prove a
      same-head clean-to-`CHANGES_REQUESTED` transition during final evidence
      collection cannot reach merge readiness in either producer.
  16. `tests/test_watch_owned_pr.py` and `tests/test_pr_watcher.py` prove a head
      move during the final review stability recheck cannot reach ready in
      either producer.
  17. `docs/OVERNIGHT_ARC_WORKFLOW.md` requires complete exact-head Codex review
      evidence for documentation-only and code PRs alike.
  18. `tests/test_watch_owned_pr.py` proves deletion of the sole accepted clean
      top-level Codex comment during final evidence collection cannot reach
      `MERGE-READY`.
  19. `tests/test_codex_wake_bridge.py` and
      `tests/test_report_pr_watcher_state.py` prove a producer snapshot carrying
      both an exact-head change request and optional pending checks remains
      actionable in every consumer rather than being downgraded to pending.
- Reachability proof: the real shell/Python watcher entrypoints consume mocked
  GitHub snapshots in their existing focused test suites; observable output is
  `MERGE-READY` versus pending/actionable and ready versus pending/attention
  state/report buckets.
- Affected surfaces: builder and overnight instructions, session-state handoff,
  shell watcher, Python watcher, wake-bridge readiness validation, reporter
  classification, their focused tests, and the pre-push cross-session drift
  audit.
- Risk areas: premature merge readiness, suppressed actionable review/check/
  conflict evidence, stale/incomplete review evidence, mismatched wake
  authorization, accidental model polling, and watcher merge authority.
- Reviewer rules triggered: R1, R2, R5, R6, R8, R10, R12, R13, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: GitHub formal review/comment evidence -> shell initial
  watcher decision.
  - Complete exact-head formal sequence ending in `COMMENTED`/`APPROVED`:
    intentionally changed to one effective clean attestation;
    requested-then-clean is accepted.
  - Complete exact-head formal sequence ending in `CHANGES_REQUESTED`:
    intentionally changed to actionable even when aggregate `reviewDecision` is
    empty; clean-then-requested is blocked.
  - No exact-head formal review plus one valid exact-head clean top-level Codex
    comment: preserved as a clean attestation.
  - Missing, wrong-author, stale-head, incomplete, or malformed review evidence:
    rejected to review-pending and never merge-ready.
- Boundary path/seam: final formal review snapshot -> later thread/check/merge
  snapshot -> shell final watcher decision.
  - Latest effective formal review clean with every later actionable fact clear:
    preserved as merge-ready.
  - Latest effective formal review requests changes: intentionally changed to
    actionable independent of aggregate decision.
  - Attestation disappears with no actionable fact: preserved as review-pending.
  - Attestation disappears while a thread/check/conflict becomes actionable:
    intentionally changed to actionable before review-pending.
  - Head remains the armed SHA through all final evidence collection: preserved
    as eligible for final classification.
  - Head changes while final evidence is collected: intentionally changed to
    `HEAD-MOVED`, never merge-ready or actionable on the mixed snapshot.
  - Head stays fixed but the complete exact-head formal-review snapshot changes:
    intentionally changed to unstable/incomplete review evidence and never
    merge-ready.
  - A paginated review/comment response identifies any head other than the armed
    SHA: intentionally rejected as incomplete evidence and never merge-ready.
  - The sole accepted clean top-level Codex comment disappears after the first
    final snapshot: intentionally treated as changed review evidence and never
    merge-ready.
- Boundary path/seam: paginated GitHub reviews -> Python version-1 readiness
  proof -> Python watcher state.
  - Complete exact-head sequence ending clean: intentionally changed to
    `codex_head_review_count >= 1`, `codex_changes_requested=false`, and otherwise
    eligible for ready.
  - Complete exact-head sequence ending requested: intentionally changed to
    `codex_changes_requested=true` and attention.
  - No exact-head attestation: intentionally changed to pending.
  - Missing/malformed submission timestamp or incomplete pagination: rejected
    to attention with incomplete/error evidence.
- Boundary path/seam: version-1 readiness proof -> wake bridge classification.
  - Scheduled source plus complete matching proof, positive attestation, literal
    `codex_changes_requested=false`, and clean checks/threads/merge:
    intentionally changed to scheduled-ready, still subject to live merge guards.
  - Scheduled source plus missing/malformed/true change-request evidence or any
    contradictory proof: rejected to attention with no merge authority.
  - Event source with any watcher state: preserved as event attention/no-op and
    cannot consume scheduled-ready-only authorization.
- Boundary path/seam: version-1 readiness proof -> reporter classification.
  - Complete safe proof: intentionally changed to ready.
  - Missing/malformed/true review evidence: intentionally changed to attention
    with explicit readiness blockers.
- Boundary path/seam: producer state plus optional check status -> shared
  consumer classification.
  - Closed producer or live PR state: preserved as closed/stale.
  - Actionable producer state with or without optional pending checks:
    intentionally classified as attention before pending.
  - Non-actionable pending state or optional pending checks: preserved as
    pending.
  - Ready producer state: preserved only when the version-1 proof independently
    validates; contradictory proof remains attention.
- Boundary path/seam: activation source -> active-builder merge instruction.
  - Scheduled-ready activation matching recorded authorization: preserved as
    eligible to run live merge guards, not itself authorized to merge.
  - Immediate, manual, event, unknown, or mismatched activation: rejected from
    scheduled-ready-only merge authorization.
- Boundary path/seam: explicit local body -> current-PR body audit.
  - Valid explicit body plus stale same-PR GitHub body: intentionally changed to
    use the explicit body as current authority.
  - Peer PR bodies and path/lane collisions: preserved and still audited.

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

- Consumer snapshot state membership is **CLOSED** and **DERIVED** from the
  producer states returned by `scripts/pr_watcher.py:_classify`, plus the local
  `review_changed` receipt state. `classify_snapshot_state` is the single
  consumer choke point. An out-of-vocabulary producer state returns `other`;
  the wake bridge then defaults it to attention and the reporter never promotes
  it to ready.
- Watcher JSON representation is **OPEN** and interpreted at the shared choke
  point. Top-level actionable diagnostics default to attention, explicit live
  terminal state maps to closed/stale, actionable producer state precedes every
  optional-check representation, truthy pending representations default to
  pending, and only a recognized ready state with a valid version-1 proof can
  reach ready. `test_codex_wake_bridge_snapshot_state_property` generates state
  tokens x pending containers x blocker-key families x live GitHub states and
  checks both a contract-derived semantic oracle and representation parity.
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
- `docs/OVERNIGHT_ARC_WORKFLOW.md`
- `docs/SESSION_STATE_TEMPLATE.md`
- `docs/long_running_session_watcher_handoff.md`
- `plans/PR-Agent-Efficiency-Rules.md`
- `scripts/audit_pr_session_drift.py`
- `scripts/codex_wake_bridge.py`
- `scripts/pr_watcher.py`
- `scripts/report_pr_watcher_state.py`
- `scripts/watch_owned_pr.sh`
- `tests/test_audit_pr_session_drift.py`
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

The shell watcher's final readiness pass collects the complete review
attestation first and then reads review threads, review decision, and merge
state. That ordering prevents a newly submitted review from being accepted
without observing the threads created with it. Definite negative merge states
and all other final-read actionable facts are classified before missing-review
pending states. Both review paginators preserve an exact-head Codex
`CHANGES_REQUESTED` signal independently of the aggregate PR decision; the
producers classify it as actionable/attention and the versioned proof validator
rejects any snapshot carrying it. The paginators reduce ordered exact-head
formal reviews to the connector's latest submitted state, so a later clean
review clears an earlier request while the opposite order remains blocking.
After every final review, thread, merge, and check fact is collected, the shell
watcher re-reads the PR head and rejects the whole observation as `HEAD-MOVED`
when it no longer matches the armed SHA. The Python producer already performs
the equivalent post-review head comparison.
Both producers also collect the complete exact-head formal-review snapshot on
both sides of their final thread/check/head observations. A changed snapshot is
not classified from either side; it fails closed as unstable review evidence.
Every review and clean-comment page also returns `headRefOid`, which must equal
the armed head before the page contributes evidence. The overnight workflow
applies that same exact-head requirement to documentation-only PRs rather than
treating a green reconciliation check as review presence.
The shell's final stability recheck re-fetches clean top-level Codex comments as
well as formal reviews and compares both accepted-source counts/snapshots before
readiness.
The wake bridge owns one shared semantic classifier for stored watcher
snapshots. Both it and the state reporter apply the same actionable-diagnostic,
authoritative-closed, actionable-producer, pending, then proof-validated-ready
precedence, so optional pending checks cannot suppress a producer's actionable
review state. The reporter supplies its live GitHub state to that classifier;
the bridge continues to reject a stored ready/closed contradiction through the
version-1 proof validator.

The instruction set distinguishes active model turns from external state
collectors: external timers/webhooks may wake a session, but the session takes
one snapshot and yields again on unchanged/pending state. Merge authorization
continues to be conditioned on its recorded activation source.

The session-drift audit treats `--current-pr-body-file` as the intended body for
the current branch and does not append stale same-PR GitHub-body errors after
that file passes. It still loads GitHub PRs and evaluates every peer PR for path
and lane overlap.

## Intentional

- A missing Codex review is pending only when the same snapshot carries no
  actionable failure, conflict, unresolved thread, or exact-head change request.
- Incomplete review API data or an open changes-requested decision is attention,
  because treating uncertainty as ready would weaken the merge gate.
- Local non-model watchers may continue polling GitHub; this slice removes model
  polling, not deterministic state collection.
- The local body override applies only to the current PR identified by branch or
  head. It does not suppress peer-PR collision checks.

## Deferred

- Attention-handoff convenience text does not currently surface
  `codex_reviews_error`. The producer still classifies malformed/incomplete
  review collection as attention and stores the diagnostic in its status JSON,
  so this is deferred under the operator's no-more-hardening instruction unless
  an observed operator handoff cannot identify the cause.
- The shell's final review-stability pagination does not add a repeated-cursor
  guard or page cap. A syntactically valid GitHub response that repeats a
  nonempty cursor can prevent the watcher from returning, but cannot produce a
  false ready result; this resource-hardening case is deferred under the
  operator's no-more-hardening instruction.

Parked hardening: attention-handoff diagnostic projection and shell final-review
recheck pagination bounds; no false-readiness behavior is deferred.

## Verification

- Fail-first: same-head clean-to-change-request review transitions reached ready
  in both producers (`2 failed`) before stability revalidation.
- Focused same-head review-transition regressions - `2 passed`.
- Fail-first: a head move during each producer's final review recheck still
  reached ready (`2 failed`).
- Focused head-bound final-review recheck regressions - `2 passed`.
- Fail-first: deleting the sole accepted clean top-level Codex comment during
  the shell's final evidence interval still emitted `MERGE-READY` (`1 failed`).
- Focused clean-comment disappearance regression - `1 passed`.
- Fail-first actionable-state consumer regressions - `2 failed` because both
  consumers downgraded attention plus an optional pending check to pending.
- Focused actionable-state consumer regressions - `2 passed`.
- `uv run pytest -q tests/test_codex_wake_bridge.py
  tests/test_report_pr_watcher_state.py` - `80 passed`.
- `uv run pytest -q tests/test_watch_owned_pr.py tests/test_pr_watcher.py
  tests/test_codex_wake_bridge.py tests/test_report_pr_watcher_state.py` -
  `200 passed`.
- `uv run pytest -q tests/test_watch_owned_pr.py` - `37 passed`.
- `uv run pytest -q tests/test_watch_owned_pr.py tests/test_pr_watcher.py` -
  `119 passed` with head-bound review/comment pages in both producers.
- `uv run pytest -q tests/test_watch_owned_pr.py tests/test_pr_watcher.py` -
  `117 passed` after bracketing final evidence with complete review snapshots.
- Fail-first: a head transition during final shell evidence collection still
  emitted `MERGE-READY` (`1 failed`) before moving the head check.
- Focused before/during-final-evidence head-move regressions - `2 passed`.
- `uv run pytest -q tests/test_watch_owned_pr.py` - `34 passed` after the
  post-collection head revalidation fix.
- Fail-first: four targeted regressions failed in the declared readiness class
  before implementation (`4 failed`).
- Fail-first: the final-read interleaving and both conflict-precedence probes
  failed before this review-round implementation (`3 failed`).
- Fail-first: the final-read actionable precedence, exact-head change-request
  propagation, and proof-validation probes failed before this implementation
  (`6 failed, 31 passed`).
- Focused regressions for those two root classes - `38 passed`.
- Fail-first: effective-review ordering and missing-submission-time probes -
  `4 failed, 2 passed` before implementation.
- Focused effective-review regressions - `9 passed`.
- `uv run pytest -q tests/test_watch_owned_pr.py tests/test_pr_watcher.py` -
  `114 passed` after the effective-review fix.
- `uv run pytest -q tests/test_watch_owned_pr.py tests/test_pr_watcher.py
  tests/test_codex_wake_bridge.py tests/test_report_pr_watcher_state.py` -
  `176 passed`.
- `uv run pytest -q tests/test_watch_owned_pr.py tests/test_pr_watcher.py` -
  `104 passed`.
- `uv run pytest -q tests/test_codex_wake_bridge.py
  tests/test_report_pr_watcher_state.py` - `68 passed`.
- `uv run ruff check scripts/codex_wake_bridge.py scripts/pr_watcher.py
  tests/test_codex_wake_bridge.py tests/test_pr_watcher.py
  tests/test_report_pr_watcher_state.py tests/test_watch_owned_pr.py` - all
  checks passed.
- `python scripts/check_diff_budget.py --additions 1216 --body-file
  /tmp/atlas-agent-efficiency-pr-body.md` - override accepted.
- `uv run ruff format --check ...` - not applied: it would bulk-reformat all
  six existing touched Python files, including unrelated baseline formatting.
- `bash -n scripts/watch_owned_pr.sh` - exit 0.
- `python scripts/audit_pr_watcher_safety.py --repo-only` - OK; watcher
  docs/config/source grant no merge authority.
- `python scripts/check_guard_class_closure.py --base origin/main --strict` -
  OK; no guard-shaped change without a property test.
- Targeted mandatory-doc audit - `stale model polling directives: 0`.
- Canonical handoff review-proof audit - all four enforced review-attestation
  fields documented (`4/4`).
- `git diff --check` - exit 0.
- `uv run pytest -q tests/test_audit_pr_session_drift.py` - `50 passed`.
- `uv run ruff check scripts/audit_pr_session_drift.py
  tests/test_audit_pr_session_drift.py` - all checks passed.
- At push, `scripts/push_pr.sh` owns the single local review bundle; it is not
  duplicated manually.

## Estimated diff size

| File | LOC |
|---|---:|
| `AGENTS.md` | 86 |
| `CLAUDE.md` | 18 |
| `docs/OVERNIGHT_ARC_WORKFLOW.md` | 10 |
| `docs/SESSION_STATE_TEMPLATE.md` | 14 |
| `docs/long_running_session_watcher_handoff.md` | 87 |
| `plans/PR-Agent-Efficiency-Rules.md` | 522 |
| `scripts/audit_pr_session_drift.py` | 27 |
| `scripts/codex_wake_bridge.py` | 58 |
| `scripts/pr_watcher.py` | 142 |
| `scripts/report_pr_watcher_state.py` | 19 |
| `scripts/watch_owned_pr.sh` | 183 |
| `tests/test_audit_pr_session_drift.py` | 58 |
| `tests/test_codex_wake_bridge.py` | 148 |
| `tests/test_pr_watcher.py` | 257 |
| `tests/test_report_pr_watcher_state.py` | 39 |
| `tests/test_watch_owned_pr.py` | 246 |
| **Total** | **1914** |

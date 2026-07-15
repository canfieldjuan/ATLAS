# PR-Public-Lane-Contract

## Why this slice exists

Issue #2104 promotes the process gates that keep builder work inside its
assigned lane. The recent Session-Lane Admission plan established an
`Ownership lane:` in new plans, but the independently machine-consumed full PR
body still carries only `Slice phase:`. As a result, the body used to discover
peer lanes has no required lane declaration, and the local drift audit can
accept a lane-looking line from outside the plan Scope section. This is the
small public-metadata prerequisite for the later trusted `session-lane` gate;
it is a workflow/process slice justified by #2104's lane-collision risk.

The reviewer-established boundary cases make this slice exceed the 400-LOC
target. The canonical body parser, fence-aware structural walkers, and their
failure-branch fixtures must land together: splitting them would either leave
a known prose/fence bypass or claim a lane contract that its real entrypoints
cannot satisfy. The overage is regression proof, not adjacent cleanup.

### Problem-derived contract

- Root cause: The plan and PR-body contracts describe related but independently
  parsed metadata. `scripts/audit_pr_body.py` validates a `Slice phase:` but
  has no `Ownership lane:` requirement; `scripts/audit_pr_session_drift.py`
  accepts every lane-shaped line in a new plan instead of proving there is one
  real declaration in Scope. Therefore a current PR body cannot be bound to
  its plan lane, and prose or fenced examples can produce a false collision
  signal or mask a missing declaration. In particular, a heading inside a
  fenced example changes Scope state, the plan phase extractor still accepts
  fenced headings and metadata through a regex section match, and the current
  body matcher accepts a matching lane from body prose rather than the
  canonical header.
- Review-round evidence: The initial fence correction recognized only
  backticks, allowed phase/lane metadata after the why paragraph, and accepted
  a later Scope lane as the declaration. Those are one invariant failure:
  collision metadata must be parsed only from the unfenced canonical lead
  blocks, not merely from a line matching the right label.
- Correct fix must touch/change: The full-body parser must require exactly one
  canonical `Ownership lane:` directly after `Slice phase:`. The drift parser
  must require exactly one canonical non-fenced lane declaration in the Scope
  section, ignore fenced headings and metadata while finding both Scope
  declarations, reject declarations outside Scope, and compare only the
  canonical current PR-body header lane to that plan lane. The body audit must
  likewise ignore fenced text when locating its lead metadata and required
  sections. Regression tests must cover valid matching metadata and missing,
  duplicate, malformed, misplaced, fenced, and mismatched metadata. The
  `open_pr`, `push_pr`, and local-review valid-body fixtures must use that new
  shape because those entrypoints invoke the changed parser. `AGENTS.md` must
  publish the new full-body shape.
- Reviewed-head completion required: Fence walkers must recognize both
  backtick and tilde Markdown fences; body metadata must occupy the canonical
  lead sequence immediately after `Plan:`; and the Scope lane must be the
  first non-fenced, non-empty Scope line with the Scope phase read from the
  canonical lead declaration. The fix must preserve intervening fence
  boundaries rather than making distant lines adjacent, and a shorter nested
  fence delimiter must not close a longer outer fence. The maturity ratchet
  must remain at its existing score through behavior-preserving guarded
  access, and the Unit Gate baseline must only shrink for now-passing
  session-drift entries; it must never absorb a regression. The
  machine-consumed PR body must carry the required diff-budget override
  explaining the already-justified overage.
- Must not change: Docs-only and Dependabot exemptions; existing Plan and Slice
  phase validation; advisory peer-file overlap reporting; branch-protection
  settings; trusted workflow topology; reviewer-status publication; private
  session-state authorization; product behavior; and other sessions' PRs.

## Scope (this PR)

Ownership lane: dev-workflow/process-gate-enrollment
Slice phase: Workflow/process

1. Require the full PR-body `Ownership lane:` line immediately after `Slice
   phase:` and bind it exactly to the sole Scope lane in the branch-added plan.
2. Reject missing, duplicate, malformed, fenced, out-of-Scope, prose-only,
   and mismatched declarations with focused failure-branch tests.
3. Update the wrapper entrypoint fixtures that intentionally pass a valid full
   body through the changed parser.
4. Preserve Scope state across fenced example headings, ignore fenced metadata
   in all plan and body structural scans, and ignore peer-body prose lanes
   unless they appear in the canonical header position.
5. Complete the same canonical-lead rule for tilde fences, body metadata order,
   and the first Scope line; prove those held-out forms with focused tests.
6. Clear the directly caused CI ratchets without waiving findings or adding a
   Unit Gate baseline entry.

### Review Contract

- Acceptance criteria:
  - [ ] A full human body has one canonical lane directly after its phase line.
  - [ ] A changed plan has one canonical, non-fenced lane inside Scope only.
  - [ ] The current body lane and plan Scope lane must match exactly.
  - [ ] Headings or lane-looking text inside examples/prose cannot satisfy
        either parser.
  - [ ] Fenced phase lines and required-section headings cannot satisfy either
        parser's structural contract.
  - [ ] Both Markdown fence delimiters and every canonical lead-block position
        obey the same non-structural parsing rule.
  - [ ] Maturity and Unit Gate ratchets do not gain an unreviewed exception.
  - [ ] The open-PR and push-PR wrapper paths still accept a valid full body.
  - [ ] Docs-only and Dependabot behavior remains unchanged.
- Reachability proof: `scripts/push_pr.sh` audits the body and
  `scripts/local_pr_review.sh --current-pr-body-file` invokes the drift
  audit; a mismatched fixture must make that real local PR-preflight path fail.
- Affected surfaces: PR body contract, session-drift metadata parsing, local
  pre-push entrypoints and their valid-body fixtures, and the AGENTS template.
- Risk areas: malformed metadata becoming a false lane claim, a stale body
  passing with a different plan lane, and unintended breakage of exempt bodies.
- Reviewer rules triggered: R1, R2, R10, R14.

### Files touched

- `AGENTS.md`
- `plans/PR-Public-Lane-Contract.md`
- `scripts/audit_pr_body.py`
- `scripts/audit_pr_session_drift.py`
- `tests/test_audit_pr_body.py`
- `tests/test_audit_pr_session_drift.py`
- `tests/test_local_pr_review.py`
- `tests/test_open_pr_wrapper.py`
- `tests/test_push_pr_wrapper.py`
- `tests/unit_gate_baseline.txt`

## Mechanism

Use the existing lower-case lane grammar without normalizing aliases. The body
audit recognises exactly one header declaration, immediately following the
phase line, and treats it as metadata rather than prose. Both auditors walk
Markdown structure with fenced lines excluded, so an example cannot provide a
header, required section, Scope boundary, or phase. The drift audit then reads
current and peer lanes from the canonical header position only and validates
the current body lane against the plan's sole Scope lane before a PR exists or
after it is opened. Other open PRs remain advisory inputs until the later
`session-lane` workflow is introduced and enrolled. The wrapper and local-review
tests supply the same valid header shape so they continue to prove each real
local publication entrypoint.

The completion pass treats both CommonMark fence delimiters as non-structural,
without removing their separator position. It reads body metadata only from
the contiguous `Plan`/phase/lane lead block, reads GitHub-loaded body phase
metadata from that same header, and reads plan Scope phase metadata from the
canonical Scope lead declaration rather than the whole Scope body.
Iterator-based guarded lookups retain the existing maturity ratchet score; the
Unit Gate ledger removes only the two session-drift entries proven stale by
this repair.

## Intentional

- This is a public collision signal, not a durable lane lease or proof that an
  ignored local session-state file authorized the work.
- Existing open human PRs are not edited or backfilled; their absent body lane
  remains non-colliding during this advisory rollout.
- The existing trusted-base body workflow gains the stricter body shape through
  its unchanged parser invocation; a distinct `session-lane` workflow belongs
  to the later producer slice.
- The local caller hint named unrelated `build_report` functions by text match;
  its real PR-body wrapper callers are covered by the updated wrapper tests.

## Deferred

- #2104 follow-up producer slice: make `session-lane` fail closed on GitHub
  metadata outages and independently visible in trusted-base CI.
- #2104 enrollment slice: source-pin the advisory contexts in branch
  protection after burn-in and legacy-PR clearance.
- #2104 reviewer slice: replace the forgeable status publisher with the
  Codex-owned `claude-review` Actions check.

Parked hardening: none.

## Verification

- Focused pytest command for the two auditors, local-review fixture, and two
  wrapper suites — 130 passed.
- Maturity-sweep structural scores: body auditor 5 and drift auditor 7, at or
  below their committed ratchet baselines.
- Full Unit Gate command confirms the two session-drift nodes are now passing
  and removed from the baseline. Local full-gate output still reports
  unrelated environment noise: nine security baseline entries pass locally but
  not in GitHub's previous hosted run, and `test_nemotron_stt` fails only in
  the full local sweep while passing alone. CI remains the final hosted
  verification after publication.
- Local PR review command with the planned current-body file against origin/main — pending final push.

## Estimated diff size

| File | LOC |
|---|---:|
| `AGENTS.md` | 1 |
| `plans/PR-Public-Lane-Contract.md` | 197 |
| `scripts/audit_pr_body.py` | 96 |
| `scripts/audit_pr_session_drift.py` | 286 |
| `tests/test_audit_pr_body.py` | 109 |
| `tests/test_audit_pr_session_drift.py` | 319 |
| `tests/test_local_pr_review.py` | 1 |
| `tests/test_open_pr_wrapper.py` | 1 |
| `tests/test_push_pr_wrapper.py` | 1 |
| `tests/unit_gate_baseline.txt` | 3 |
| **Total** | **1014** |

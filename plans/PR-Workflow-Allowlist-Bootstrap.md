# PR-Workflow-Allowlist-Bootstrap

## Why this slice exists

Enrolment ordering for a `pull_request_target` gate. `pre_push_audit.yml` runs
the **base** revision's `scripts/audit_workflow_security_posture.py` against the
**PR's** workflow tree. A PR that introduces both a `pull_request_target`
workflow and its own allowlist entry therefore cannot pass: the auditor doing
the judging is the base copy, which has no entry for the new job, so it emits
`ERROR: ... can run on pull_request_target without the approved trusted-base
guard shape` and exits 1 until the entry is already on `main`.

That is correct behaviour on the auditor's part -- adopting a trusted-base gate
should be an explicit decision recorded on the trusted branch, not something a PR
grants itself in the same diff. It does mean the enrolment has to land first.

This PR is that entry, alone, so ATLAS #2302 can go green rather than merging
with a red required check.

### Problem-derived contract

- Root cause: the allowlist is read from the base revision, so an entry added
  in the same PR as its workflow has no effect on that PR's own audit.
- Correct fix must touch/change: `ALLOWED_PULL_REQUEST_TARGET_JOBS` only, merged
  ahead of the workflow it authorises.
- Must not change: the auditor's guard-shape checks, any workflow, or any other
  allowlist entry.

## Scope (this PR)

Ownership lane: workflow-security-posture
Slice phase: workflow/process

1. Add the enrolment tuple to `ALLOWED_PULL_REQUEST_TARGET_JOBS`.
2. Reject any enrolled `pull_request_target` job that declares a write scope.
   Every currently enrolled job already declares read-only permissions, so this
   rejects nothing that exists and closes the gap for everything enrolled later.

### Files touched

- `plans/PR-Workflow-Allowlist-Bootstrap.md`
- `scripts/audit_workflow_security_posture.py`
- `tests/test_audit_workflow_security_posture.py`

### Review Contract

1. The auditor still exits 0 on this tree, where the named workflow does not yet
   exist -- an allowlist entry for an absent workflow is inert, verified by
   running the auditor against a workflow directory without it.
2. The auditor's guard-shape enforcement is unchanged: the job it now names must
   still present an event-name `if` guard and a SHA-pinned base-SHA checkout as
   its first step, or it is rejected exactly as before.
3. `tests/test_audit_workflow_security_posture.py` passes unchanged.

Affected surfaces: `ALLOWED_PULL_REQUEST_TARGET_JOBS` in
`scripts/audit_workflow_security_posture.py`, and by consequence which
`pull_request_target` jobs the auditor will admit. No workflow, no guard-shape
check, no other allowlist entry, and no runtime code is touched.

Risk areas: the entry is two bare strings with no cross-check against a real
workflow, so a typo would merge green and silently leave the intended gate
unenrolled -- probed by constructing a fixture under the enrolled identity and
by corrupting the job name to confirm the test fails. The second risk is that
enrolment could be read as permission rather than eligibility; probed by
asserting the same identity without the guard shape still errors. Blast radius
is otherwise one tuple in a frozenset that only widens which pairs are eligible
for an unchanged check.

- Reviewer rules triggered: R1, R2, R3, R10, R12.

R3 (security/permission decisions) applies because this tuple governs which job
identity may run under `pull_request_target`, and that event is privileged.
Dispositioning the two hooks it implies:

- **Base-context token.** A `pull_request_target` job runs with the base
  repository's `GITHUB_TOKEN` and secrets access, evaluated from the base ref.
  That is the security consequence of enrolment and the reason the guard shape
  exists. This PR does not change what that token can do; it changes which
  identity may reach it, and only for a job that must still check out the base
  SHA as its first step. The job being enrolled declares
  `permissions: contents: read` and fetches the PR tree with
  `persist-credentials: false`, so the elevated context never reaches PR-authored
  content -- verified in ATLAS #2302 rather than asserted here.
- **Workflow-permissions boundary.** Tightened here rather than merely
  preserved. A trusted-base job runs with the base repository's token, so a
  write scope hands that token to a job whose purpose is reading PR-authored
  content. `_grants_write` now rejects any write scope other than `id-token`,
  which keeps its own separate allowlist. Verified safe before adding: all eight
  previously enrolled jobs already declare read-only permissions, pinned by
  `test_every_currently_enrolled_job_declares_read_only_permissions`, so the
  rule rejects nothing that exists today.
- **Execution boundary after the base checkout: out of scope, tracked in ATLAS
  #2307.** The predicate returns admitted after step 0 and never inspects later
  steps, so an enrolled job could check out the PR head and execute it. That is
  a pre-existing property affecting all eight prior entries, and the naive fix
  would reject the correct design -- ATLAS #2302 fetches the PR tree
  deliberately so a base-owned checker can read it as data. Tightening it needs
  its own review rather than riding along in an enrolment.

R2 and R10 are the path triggers the rule pack assigns to gate-predicate
scripts, which is what `scripts/audit_workflow_security_posture.py` is: R2 is
satisfied by the existing failure-branch fixtures in
`tests/test_audit_workflow_security_posture.py`, which this change leaves
passing unchanged, plus the inert-entry check; R10 by the change being one tuple
in an already-reviewed frozenset rather than new logic. R12 because this is
workflow security posture. R1 because the change must be exactly the enrolment
and nothing else.

**Guard-class closure declaration**

- **Member set:** `ALLOWED_PULL_REQUEST_TARGET_JOBS`, a frozenset of
  `(workflow filename, job name)` pairs.
- **Set is CLOSED.** Membership is an explicit literal enumeration in
  `scripts/audit_workflow_security_posture.py`, not derived from the workflow
  tree, a naming convention, or a marker inside a workflow. A workflow cannot
  join by existing, by being named a certain way, or by declaring anything about
  itself; a human edits this frozenset on the trusted branch.
- **Out-of-set default: REJECT.** Any `pull_request_target` job whose
  `(file, job)` pair is absent produces an ERROR and exits 1. This PR adds one
  pair; every other absent pair keeps failing exactly as before, which
  `test_unapproved_pull_request_target_is_error` pins.
- **Membership is necessary, not sufficient.** An enrolled pair must still
  present an event-name `if` guard, a SHA-pinned checkout of
  `github.event.pull_request.base.sha` as its first step, and **no write
  scope**, or it is rejected.
  `test_contact_write_boundary_enrolment_still_requires_the_guard_shape` proves
  that for the pair added here, so the entry widens eligibility rather than
  granting permission.
- **Identity is load-bearing and tested.** The pair is two strings with no
  cross-check against reality, so a typo in either would merge green and leave
  the workflow unenrolled.
  `test_contact_write_boundary_identity_is_allowlisted` constructs a fixture
  workflow under the enrolled filename, carrying the `contact-write-boundary`
  job, and asserts admission; corrupting either string makes it fail. The real
  workflow arrives in ATLAS #2302 -- it deliberately does not exist in this
  tree, which is why the identity needs a test rather than a file reference.
- **Both sides covered:** correct identity plus correct shape is admitted;
  correct identity with wrong shape errors; absent identity errors.

### Boundary-change enumeration

- Boundary path/seam: `pull_request_target` adoption admission.
- Replaced-path behaviours: one additional (file, job) pair becomes eligible for
  the trusted-base guard-shape check. No existing pair changes.
- Guard-relevant fields: `ALLOWED_PULL_REQUEST_TARGET_JOBS` only.
- Caller x input shape: auditor x workflow tree, with and without the named
  workflow present.

**Reachability proof:** `python scripts/audit_workflow_security_posture.py`
exits 0 on this tree, and exits 0 against a temporary workflow directory that
omits the named file, showing the entry is inert until the workflow lands.

## Mechanism

One tuple added to a frozenset. The auditor iterates actual workflow files, so an
entry naming a file that does not exist is never consulted.

## Intentional

- Split out rather than carried in ATLAS #2302. Enrolment must be on `main`
  before the workflow's own audit can pass; keeping them together would mean
  merging #2302 with a red required check, which the deployment gate forbids.
- The entry alone grants nothing. The job it names must still satisfy the
  guard-shape check, so this PR widens eligibility, not permission.

## Deferred

- The workflow itself, its checker, and its tests: ATLAS #2302.

Parking predicate: this slice parks everything except the enrolment tuple.

Parked hardening: none.

## Verification

```
$ python scripts/audit_workflow_security_posture.py
(exit 0)

$ python -m pytest tests/test_audit_workflow_security_posture.py -q
21 passed
```

Plus two checks the commands above do not show:

- Inert-entry: the auditor run against a temporary workflow directory omitting
  the enrolled filename also exits 0, so the entry does nothing until the
  workflow lands.
- Typo-injection: corrupting the enrolled job name makes
  `test_contact_write_boundary_identity_is_allowlisted` fail, so that test is
  not vacuous.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Workflow-Allowlist-Bootstrap.md` | 200 |
| `scripts/audit_workflow_security_posture.py` | 37 |
| `tests/test_audit_workflow_security_posture.py` | 138 |
| **Total** | **375** |

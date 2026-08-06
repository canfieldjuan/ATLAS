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

This PR is that entry, so ATLAS #2302 can go green rather than merging with a
red required check -- plus the one precondition that adding an entry would
otherwise widen: an enrolled job must prove its token is read-only. See root
cause B below.

### Why this slice is over the 400-LOC target

534 added lines, of which **57 are production code**:

| File | LOC | What it is |
|---|---:|---|
| `plans/PR-Workflow-Allowlist-Bootstrap.md` | 252 | this document, required by the plan contract |
| `tests/test_audit_workflow_security_posture.py` | 222 | boundary table + three end-to-end fixtures |
| `scripts/audit_workflow_security_posture.py` | 57 | the actual change |

The production delta is not splittable, and the honest reason is narrower than
"all of it is necessary":

- **The tuple and the precondition are one decision.** This PR enrols a ninth
  identity into an admission predicate that fails open on omitted permissions.
  Shipping the enrolment alone widens that hole by exactly the job being added.
  Shipping the precondition alone hardens a gate nothing is yet passing through,
  and still leaves ATLAS #2302 unable to go green. Neither half is independently
  correct.
- **The test weight is the point, not padding.** The production change is a
  predicate whose entire value is where it says no, so it needs both error
  directions plus the shapes it cannot evaluate statically. That is one 12-row
  table and three fixtures; there is no smaller honest version.
- **The plan doc is contract machinery**, not slice content: 47% of the diff and
  0% of the behaviour.

What I will not claim: that this was always going to be one indivisible unit.
The first version of this PR was the enrolment tuple alone, comfortably under the
cap. The growth came from Codex's R3, which was correct. So the overage is real
review-driven scope that became indivisible once the fail-open was found -- not a
slice that was sized this way from the start.

### Problem-derived contract

This slice has two root causes, because enrolling a ninth job surfaced a
fail-open in the admission predicate that the previous eight silently relied on.

- Root cause A (ordering): the allowlist is read from the base revision, so an
  entry added in the same PR as its workflow has no effect on that PR's own
  audit.
- Root cause B (permission boundary): admission never checked the token the
  admitted job would receive. An enrolled `pull_request_target` job that
  declared a write scope -- or that declared no `permissions` block at all, and
  so inherited a repository/organisation default that is write-capable and is
  configured outside this repository -- was admitted. That hands a
  write-capable base-context token to a job whose entire purpose is reading
  PR-authored content. Adding an entry without closing this would widen the
  hole by one more identity.
- Correct fix must touch/change: `ALLOWED_PULL_REQUEST_TARGET_JOBS` (A), and
  the admission predicate in `_is_allowed_pull_request_target_job` gaining an
  explicit-read-only-permissions precondition (B), merged ahead of the workflow
  it authorises.
- Must not change: the existing guard-shape checks (event-name `if`, SHA-pinned
  base-SHA first checkout), any workflow, or any other allowlist entry. B adds a
  precondition alongside those checks; it does not alter them.

## Scope (this PR)

Ownership lane: workflow-security-posture
Slice phase: workflow/process

1. Add the enrolment tuple to `ALLOWED_PULL_REQUEST_TARGET_JOBS`.
2. Require an enrolled `pull_request_target` job to declare permissions that are
   explicitly and provably read-only. This rejects a declared write scope, and
   equally rejects an omitted block, `write-all`, a bare scalar, and any scope
   whose value is an unresolved `${{ }}` expression the auditor cannot evaluate
   statically. Stated as a positive predicate
   (`_permissions_are_explicitly_read_only`) rather than a "does it grant write"
   negative, so unrecognised shapes fall on the reject side by construction.
   All eight previously enrolled jobs already declare an explicit read-only
   block, so this rejects nothing that exists and closes the gap for everything
   enrolled later.

### Files touched

- `plans/PR-Workflow-Allowlist-Bootstrap.md`
- `scripts/audit_workflow_security_posture.py`
- `tests/test_audit_workflow_security_posture.py`

### Review Contract

1. The auditor still exits 0 on this tree, where the named workflow does not yet
   exist -- an allowlist entry for an absent workflow is inert, verified by
   running the auditor against a workflow directory without it.
2. The pre-existing guard-shape checks are unaltered: the job it now names must
   still present an event-name `if` guard and a SHA-pinned base-SHA checkout as
   its first step, or it is rejected exactly as before. The permissions
   precondition is added alongside them, not in place of them.
3. The permissions precondition rejects nothing currently enrolled.
   `test_every_currently_enrolled_job_declares_read_only_permissions` resolves
   each enrolled pair against the real workflow tree and asserts the predicate
   admits it, and fails if that loop ever checks zero workflows.
4. `tests/test_audit_workflow_security_posture.py` passes. Three positive
   fixtures were amended, not weakened: they omitted `permissions` entirely
   while every real workflow they model declares an explicit read-only block, so
   they were asserting admission for a shape that no longer exists in the repo.

Affected surfaces: `ALLOWED_PULL_REQUEST_TARGET_JOBS` and the admission
predicate `_is_allowed_pull_request_target_job` in
`scripts/audit_workflow_security_posture.py`, and by consequence which
`pull_request_target` jobs the auditor will admit. No workflow, no existing
guard-shape check, no other allowlist entry, and no runtime code is touched.

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
  preserved; this is root cause B. A trusted-base job runs with the base
  repository's token, so a write scope hands that token to a job whose purpose
  is reading PR-authored content.
  `_permissions_are_explicitly_read_only` admits only an explicit block that
  provably grants no write other than `id-token` (which keeps its own separate
  allowlist). Both sides of the boundary are probed by
  `test_permissions_read_only_predicate_boundaries`: `{contents: read}`,
  `{}`, `read-all`, and `{contents: read, id-token: write}` admit; `None`,
  `write-all`, `{contents: write}`, a bare `"read"` scalar, and a templated
  `${{ }}` value reject. The omitted-permissions case is the one that matters
  most and has its own end-to-end fixture,
  `test_enrolled_job_omitting_permissions_entirely_is_rejected`: absence is not
  evidence of read-only, because the inherited default lives in repository or
  organisation settings that this file cannot see. Verified safe before adding:
  all eight previously enrolled jobs already declare an explicit read-only
  block -- enumerated directly against the workflow tree, not assumed -- pinned
  by `test_every_currently_enrolled_job_declares_read_only_permissions`, so the
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
`tests/test_audit_workflow_security_posture.py` plus the new reject-side
fixtures and the inert-entry check; R10 by the added logic being a single
predicate over a permissions block, with both sides enumerated. R12 because this
is workflow security posture. R1 because the change must be exactly the
enrolment and the permission precondition that enrolment would otherwise widen
-- and nothing else.

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
  `github.event.pull_request.base.sha` as its first step, and **an explicit
  permissions block that provably grants no write** -- an omitted block is not
  accepted as read-only -- or it is rejected.
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

Two things. One tuple added to a frozenset -- the auditor iterates actual
workflow files, so an entry naming a file that does not exist is never
consulted. And one precondition added to the admission predicate: before an
enrolled job is admitted, its effective permissions (job block if present, else
workflow block) must be an explicit, statically provable read-only shape.

## Intentional

- Split out rather than carried in ATLAS #2302. Enrolment must be on `main`
  before the workflow's own audit can pass; keeping them together would mean
  merging #2302 with a red required check, which the deployment gate forbids.
- The entry alone grants nothing. The job it names must still satisfy the
  guard-shape check, so this PR widens eligibility, not permission.

## Deferred

- The workflow itself, its checker, and its tests: ATLAS #2302.

Parking predicate: this slice parks everything except the enrolment tuple and
the permissions precondition that enrolment would otherwise widen.

Parked hardening: the execution boundary after the base checkout (ATLAS #2307),
for the reason given above -- pre-existing across all eight prior entries, and
the naive fix would reject #2302's deliberate read-the-PR-tree-as-data design.

## Verification

```
$ python scripts/audit_workflow_security_posture.py
(exit 0)

$ python -m pytest tests/test_audit_workflow_security_posture.py -q
37 passed
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
| `plans/PR-Workflow-Allowlist-Bootstrap.md` | 284 |
| `scripts/audit_workflow_security_posture.py` | 60 |
| `tests/test_audit_workflow_security_posture.py` | 222 |
| **Total** | **566** |

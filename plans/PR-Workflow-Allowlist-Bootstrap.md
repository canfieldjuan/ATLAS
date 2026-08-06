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

Over the cap, and roughly **nine tenths of it is not behaviour**. Per-file
figures live in **Estimated diff size** at the bottom of this document and
nowhere else -- `scripts/sync_pr_plan.py` regenerates that table from the real
diff, so restating the numbers here would just create a second copy that drifts
out of agreement with the first. (It already did once: this section carried a
hand-typed total that was stale within two commits while the synced table below
was correct.)

The shape, which does not drift: one file of production code, one test file
several times its size, and a plan document about as large as the tests.

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
- **The plan doc is contract machinery**, not slice content: roughly half the
  diff and none of the behaviour.

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
  so inherited whatever the repository default happens to be -- was admitted.
  A write scope there hands a write-capable base-context token to a job whose
  entire purpose is reading PR-authored content.

  **Severity, corrected downward after checking the deployment.** This
  repository's default is `read` (evidence and source recorded under
  Boundary-change enumeration), so the omitted-permissions path was **not**
  live-exploitable. An earlier revision of this document asserted the default
  was write-capable; that was an assumption, and it was wrong. What remains is
  a genuine fail-open in the predicate, defended because the setting is
  mutable outside this repository and outside review, with nothing that would
  fail a build if it flipped. The declared-write half of B is unconditionally
  real and needs no such caveat.
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
4. Every guard-shape branch is independently proven to fire. The rejection
   fixtures now carry read-only permissions and violate exactly one element each
   (missing `if`, unpinned checkout, wrong checkout ref, no steps), with a
   full-shape control proving the builder is not producing something
   universally rejected. Before this, those fixtures omitted `permissions`, so
   the new precondition rejected them first and both tests stayed green with the
   event-name and checkout checks deleted -- verified by deleting both and
   watching the suite still pass.
5. `tests/test_audit_workflow_security_posture.py` passes. Three positive
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

- Reviewer rules triggered: R1, R2, R3, R10, R12, R13.

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

**Guard-class closure declaration -- `_READ_ONLY_PERMISSION_VALUES`**

A second, separate member set, because it decides admission independently of the
identity allowlist above.

- **Member set:** `_READ_ONLY_PERMISSION_VALUES = {"read", "none"}` -- the scope
  values that count as granting no write.
- **Set is CLOSED and ENUMERATED**, from GitHub's own documented vocabulary for
  a `permissions` scope value, which is exactly `read` / `write` / `none`. It is
  not derived from the workflow tree and nothing joins by appearing there.
- **Out-of-set default: REJECT.** The predicate is `all(value in <set>)`, so any
  value that is not literally `read` or `none` fails the check. That is the
  point: it covers `write` (the known bad value) and equally covers values this
  auditor cannot evaluate statically -- an unresolved `${{ }}` expression, a
  YAML-coerced `True`, a typo like `raed`. A "does it say write" formulation
  would have admitted all three.
- **`id-token` is excluded by key, not by value**, because OIDC has its own
  separate allowlist and check (`_permissions_write_oidc`,
  `ALLOWED_ID_TOKEN_JOB`). Excluding it here would otherwise double-govern one
  scope from two places that can disagree.
- **Two non-dict shapes are handled before the set is consulted:** `read-all`
  admits (explicit and provably read-only), everything else non-dict rejects --
  which is what makes `write-all`, a bare scalar, and an omitted block (`None`)
  all fail closed.
- **Both sides covered:** `test_permissions_read_only_predicate_boundaries` is a
  12-row table over exactly these cases, six admitting and six rejecting.

### Boundary-change enumeration

- Boundary path/seam: `pull_request_target` adoption admission, in
  `_is_allowed_pull_request_target_job`.
- Replaced-path behaviours, two of them:
  1. One additional (file, job) pair becomes eligible for the trusted-base
     guard-shape check. No existing pair changes.
  2. **Admission now depends on effective permissions for every non-canonical
     allowlisted pair** -- that is all eight prior entries plus the new one. The
     `.github/workflows/review_contract.yml` / `review-contract` pair is
     unaffected, because it
     returns earlier through the canonical-text comparison and never reaches
     this branch.
- Guard-relevant fields: `ALLOWED_PULL_REQUEST_TARGET_JOBS`, plus the workflow
  block `permissions` and the job block `permissions`, and the **precedence
  between them** -- job-level wins when the key is present, otherwise the
  workflow block is used. Presence, not truthiness: a job declaring
  `permissions: {}` overrides a permissive workflow block rather than falling
  back to it.
- Caller x input shape:
  - auditor x workflow tree, with and without the named workflow present
    (inert-entry check);
  - job-level permissions present x workflow-level absent;
  - job-level absent x workflow-level present (the shape all eight prior entries
    actually use);
  - both absent (rejects -- `test_enrolled_job_omitting_permissions_entirely_is_rejected`);
  - both present and disagreeing (job wins --
    `test_enrolled_job_with_job_level_write_permission_is_rejected` puts a write
    scope on the job under a read-only workflow block and asserts rejection).

**Deployed default-config evidence (the omitted-permissions path).** Recorded
rather than left as "configured externally", and it corrects this PR's own
framing:

- Value, read today: `default_workflow_permissions: "read"`,
  `can_approve_pull_request_reviews: false`.
- Source: `GET /repos/canfieldjuan/ATLAS/actions/permissions/workflow`, i.e.
  Settings -> Actions -> General -> Workflow permissions.
- No policy layer above it: `canfieldjuan` is a **User** account, not an
  organization, so there is no org-level default that could override or tighten
  the repository setting.
- **Consequence, stated against interest:** the omitted-permissions path was
  **not** live-exploitable on this repository. Root cause B as first written
  said an omitted block "inherits a write-capable default"; on the deployed
  configuration it inherits a read-only one. The fail-open was real in the
  predicate and absent in the deployment.
- Why the guard is still correct: the value is a repository setting, changeable
  in one click, outside this repository, outside code review, and with no
  mechanism that would fail a build if it flipped. `read` has been GitHub's
  default for new repositories only since 2023; older repositories and any
  future fork or transfer can carry `write`. The guard makes admission depend on
  something the PR states explicitly rather than on a mutable setting nobody
  reviews -- defence in depth, not an exploited hole.

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
| `plans/PR-Workflow-Allowlist-Bootstrap.md` | 375 |
| `scripts/audit_workflow_security_posture.py` | 60 |
| `tests/test_audit_workflow_security_posture.py` | 302 |
| **Total** | **737** |

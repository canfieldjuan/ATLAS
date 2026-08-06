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

1. Add `("contact_write_boundary.yml", "contact-write-boundary")` to
   `ALLOWED_PULL_REQUEST_TARGET_JOBS`.

### Files touched

- `plans/PR-Workflow-Allowlist-Bootstrap.md`
- `scripts/audit_workflow_security_posture.py`

### Review Contract

1. The auditor still exits 0 on this tree, where the named workflow does not yet
   exist -- an allowlist entry for an absent workflow is inert, verified by
   running the auditor against a workflow directory without it.
2. The auditor's guard-shape enforcement is unchanged: the job it now names must
   still present an event-name `if` guard and a SHA-pinned base-SHA checkout as
   its first step, or it is rejected exactly as before.
3. `tests/test_audit_workflow_security_posture.py` passes unchanged.

- Reviewer rules triggered: R1, R2, R10, R12.

R2 and R10 are the path triggers the rule pack assigns to gate-predicate
scripts, which is what `scripts/audit_workflow_security_posture.py` is: R2 is
satisfied by the existing failure-branch fixtures in
`tests/test_audit_workflow_security_posture.py`, which this change leaves
passing unchanged, plus the inert-entry check; R10 by the change being one tuple
in an already-reviewed frozenset rather than new logic. R12 because this is
workflow security posture. R1 because the change must be exactly the enrolment
and nothing else.

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
19 passed
```

Plus an inert-entry check: the auditor run against a temporary workflow
directory omitting `contact_write_boundary.yml` also exits 0.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Workflow-Allowlist-Bootstrap.md` | 116 |
| `scripts/audit_workflow_security_posture.py` | 1 |
| **Total** | **117** |

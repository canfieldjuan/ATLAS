# PR-Incident-Response-Runbook

## Why this slice exists

Issue #1656 M6 calls for an incident-response playbook covering severity
definitions, ownership, communications, credential rotation, money-path incident
types, and postmortems. #1812 added the public disclosure path, but reports still
land in a policy vacuum: `SECURITY.md` tells researchers where to report without
linking to an operator runbook for triage and response.

Root cause: Atlas had disclosure intake metadata but no first-party incident
response operating document. This PR fixes that root for the Atlas repo by adding
the runbook, linking it from the security policy, and testing the expected doc
contract in CI.

## Scope (this PR)

Ownership lane: security/hardening-1656
Slice phase: Production hardening

1. Add `docs/INCIDENT_RESPONSE.md` with SEV0-SEV3 definitions, ownership,
   communications cadence, credential-rotation flow, paid-funnel incident types,
   and a postmortem template.
2. Link the runbook from `SECURITY.md` so the disclosure policy points operators
   to the response process.
3. Add a focused docs contract test and enroll it in a small GitHub Actions
   workflow.

### Review Contract

- Acceptance criteria:
  - [ ] `docs/INCIDENT_RESPONSE.md` exists and covers SEV0-SEV3, owners,
        communications, credential rotation, paid-funnel incident types, and a
        postmortem template.
  - [ ] `SECURITY.md` links to the incident-response runbook.
  - [ ] The docs contract test fails if the link or required runbook sections
        disappear.
  - [ ] The new docs contract test is CI-enrolled.
- Affected surfaces: security policy docs, incident-response runbook, docs-only
  CI check.
- Risk areas: overpromising operational SLAs, drift between emitted incident
  names and the runbook.
- Reviewer rules triggered: R1, R2, R5, R11, R12, R14.

### Files touched

- `.github/workflows/atlas_security_policy_docs_checks.yml`
- `SECURITY.md`
- `docs/INCIDENT_RESPONSE.md`
- `plans/PR-Incident-Response-Runbook.md`
- `tests/test_security_policy_docs.py`

## Mechanism

The runbook is a repo-local markdown document. `SECURITY.md` links to it from
the response-targets section, keeping public disclosure intake and internal
operator response connected. The docs test reads both markdown files and asserts
the link, required headings, SEV labels, paid-funnel incident names from the
current code, and postmortem template markers. The GitHub Actions workflow runs
for the security policy, runbook, test, workflow itself, and paid-funnel emitter
paths that can add or rename incident types.

## Intentional

- This PR does not implement real alert delivery (M3), CVE remediation labels
  (M9), or provider-side credential rotation (H3). The runbook names those flows
  without claiming they are fully automated.
- This PR stays Atlas-only; the portfolio companion issue can mirror or link to
  the same response process separately.

## Deferred

- #1656 M3 real alert delivery, M9 CVE remediation SLA and labels, H3 provider
  credential rotation, M1 RLS, M2 JSONB encryption, M5 scanner ratcheting, M7
  structured logging, and H5 blog-admin authz remain separate slices.

Parked hardening: none.

## Verification

- python -m unittest tests.test_security_policy_docs - 3 tests passed
- bash scripts/local_pr_review.sh --current-pr-body-file
  tmp/pr-incident-response-runbook-body.md - passed

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_security_policy_docs_checks.yml` | 39 |
| `SECURITY.md` | 5 |
| `docs/INCIDENT_RESPONSE.md` | 143 |
| `plans/PR-Incident-Response-Runbook.md` | 94 |
| `tests/test_security_policy_docs.py` | 99 |
| **Total** | **380** |

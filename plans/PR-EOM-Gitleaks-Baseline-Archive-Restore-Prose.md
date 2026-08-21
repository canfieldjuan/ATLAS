# PR-EOM-Gitleaks-Baseline-Archive-Restore-Prose

## Why this slice exists

ATLAS #2465 contains two historical Gitleaks false positives in a plan document.
The required PR scanner reports their exact, redacted fingerprints. The source
lines describe receipt metadata; they are not credentials. A baseline update is
needed because the target commit is already part of the published #2465 history.

### Problem-derived contract

- Root cause: two scanner false positives in published documentation history
  block #2465.
- Correct fix must change: append only the two exact scanner-generated,
  redacted fingerprint records to the trusted Gitleaks baseline and document
  the evidence.
- Must not change: product code, scanner rules, workflow behavior, ignore-list
  behavior, or any trusted-base baseline member.

## Scope (this PR)

Ownership lane: security/gitleaks-baseline-archive-restore-prose
Slice phase: Vertical slice
Max files: 2

1. Append these two exact fingerprints:
   - `05bf97e8085f76cbd31f91291ed4a569934aaf3a:plans/PR-EOM-Contact-Archive-Restore.md:generic-api-key:72`
   - `05bf97e8085f76cbd31f91291ed4a569934aaf3a:plans/PR-EOM-Contact-Archive-Restore.md:generic-api-key:181`
2. Preserve every trusted-base fingerprint and record the scan evidence.

### Review Contract

- Acceptance criteria:
  - The two additions match the redacted output from #2465's required
    Gitleaks scan.
  - The candidate baseline contains all 24 trusted-base members plus exactly
    these two additions.
  - A scanner run without the candidate reports the two #2465 fingerprints;
    the same run with the candidate reports none.
  - No other source, rule, workflow, or ignore-list file changes.
- Reachability proof: documentation/baseline-only change. Its observable effect
  is that #2465's required Gitleaks scan can suppress only these known
  historical false positives.
- Affected surface: `docs/security/gitleaks-baseline.json`, consumed by the
  trusted security workflow.
- Risk area: masking a real secret. Mitigation: exact redacted scanner records,
  closed membership, trusted-base preservation, and the required scan proof.
- Reviewer rules: R2 security-sensitive baseline change.

### Guard-class closure

- Class: CLOSED.
- Reviewed snapshot: the 24-member trusted-main baseline at
  `280c8e91fcabbb11320cc152caf4a76628aee456`, plus exactly the two
  fingerprints listed in this plan.
- Membership source: redacted Gitleaks output from #2465's required PR scan.
- Out-of-set behavior: any fingerprint not in this 26-member snapshot remains
  unbaselined and is reported by the required PR scan; additions require a
  separate controlled baseline-rotation review.

### Boundary-change enumeration

N/A - no API, schema, deployment, or runtime boundary changes.

### Deployed-config probing

N/A - the workflow and deployment configuration remain unchanged.

### Files touched

- `docs/security/gitleaks-baseline.json`
- `plans/PR-EOM-Gitleaks-Baseline-Archive-Restore-Prose.md`

## Mechanism

The required PR workflow reads the baseline from trusted `main` and supplies
it to Gitleaks while scanning the candidate's commit range. Gitleaks suppresses
only findings whose complete fingerprint is in that baseline. After this slice
merges, #2465's two documented false positives match the two added records;
other findings remain visible.

## Intentional

- Use the reviewed baseline route, not `.gitleaksignore`.
- Keep the branch history clean of the scanner's trigger-shaped prose so this
  baseline PR does not create findings of its own.
- Keep the target #2465 branch unchanged; its historical commits are the
  purpose of this separate baseline slice.

## Deferred

None.

Parked hardening: none.

## Verification

- JSON parses and baseline membership is main's 24 records plus the two listed
  records.
- The baseline-rotation guard accepts the labeled, two-file candidate.
- The plan synchronizer derives files and LOC from the final diff.
- GitHub's required Gitleaks PR scan validates the clean rebuilt history; the
  baseline-growth guard validates scope and preservation.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/security/gitleaks-baseline.json` | 2 |
| `plans/PR-EOM-Gitleaks-Baseline-Archive-Restore-Prose.md` | 111 |
| **Total** | **113** |

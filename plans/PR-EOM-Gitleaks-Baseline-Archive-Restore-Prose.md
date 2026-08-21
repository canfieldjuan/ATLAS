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
- Reviewer rules: R2, R3, R12, R14 — test evidence, suppression-boundary
  security, required CI behavior, and codebase verification.

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

Parking predicate: this slice parks scanner-rule or workflow changes, any
additional baseline member, and baseline-policy hardening not required to admit
these two reviewed historical false positives; each belongs to a separate
controlled security-rotation slice.

Parked hardening: none.

## Verification

- `jq -e "length == 26" docs/security/gitleaks-baseline.json` — passed;
  result `true` confirms the candidate contains 26 records.
- `python scripts/check_gitleaks_baseline_rotation.py --base-ref origin/main
  --head-ref HEAD --labels-json "[\"security-rotation\"]"` — passed; the labeled
  candidate admits the two-file rotation without trusted-baseline removal.
- `python scripts/sync_pr_plan.py
  plans/PR-EOM-Gitleaks-Baseline-Archive-Restore-Prose.md origin/main --check`
  — passed after this plan update.
- `python scripts/audit_plan_doc.py
  plans/PR-EOM-Gitleaks-Baseline-Archive-Restore-Prose.md` — passed.
- `git diff --check origin/main...HEAD` — passed.
- #2465 comparison, pinned workflow image and range
  `origin/main..origin/claude/pr-eom-contact-archive-restore`:

  ```bash
  docker run --rm -v /home/juan-canfield/Desktop/Atlas/.git:/home/juan-canfield/Desktop/Atlas/.git:ro -v "/home/juan-canfield/Desktop/01 - Effingham Office Maids/atlas-2466-repair-worktree:/home/juan-canfield/Desktop/01 - Effingham Office Maids/atlas-2466-repair-worktree:ro" -v /home/juan-canfield/Desktop/Atlas-worktrees/eom-gitleaks-baseline-repair/docs/security/gitleaks-baseline.json:/baseline.json:ro ghcr.io/gitleaks/gitleaks@sha256:c00b6bd0aeb3071cbcb79009cb16a60dd9e0a7c60e2be9ab65d25e6bc8abbb7f git --redact --verbose --log-opts="origin/main..origin/claude/pr-eom-contact-archive-restore" --baseline-path /baseline.json "/home/juan-canfield/Desktop/01 - Effingham Office Maids/atlas-2466-repair-worktree"
  docker run --rm -v /home/juan-canfield/Desktop/Atlas/.git:/home/juan-canfield/Desktop/Atlas/.git:ro -v "/home/juan-canfield/Desktop/01 - Effingham Office Maids/atlas-2466-repair-worktree:/home/juan-canfield/Desktop/01 - Effingham Office Maids/atlas-2466-repair-worktree:ro" ghcr.io/gitleaks/gitleaks@sha256:c00b6bd0aeb3071cbcb79009cb16a60dd9e0a7c60e2be9ab65d25e6bc8abbb7f git --redact --verbose --log-opts="origin/main..origin/claude/pr-eom-contact-archive-restore" --baseline-path "/home/juan-canfield/Desktop/01 - Effingham Office Maids/atlas-2466-repair-worktree/docs/security/gitleaks-baseline.json" "/home/juan-canfield/Desktop/01 - Effingham Office Maids/atlas-2466-repair-worktree"
  ```

  The trusted-main baseline command exited 1 with exactly the two listed
  redacted records across four commits; the candidate baseline command exited
  0 with no findings across the same four commits.
- GitHub required Gitleaks PR scan passed on the clean rebuilt #2466 history;
  the baseline-growth guard passed for scope and preservation.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/security/gitleaks-baseline.json` | 2 |
| `plans/PR-EOM-Gitleaks-Baseline-Archive-Restore-Prose.md` | 133 |
| **Total** | **135** |

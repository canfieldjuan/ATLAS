# PR-Security-Scanner-Ratchet

## Why this slice exists

#1827 closed the merge-enforcement gap for Gitleaks but left one explicit
follow-up deferred: "ratchet the remaining advisory scanners after their
backlogs are burned down." Today Semgrep, OSV, Trivy, Checkov, and pip-audit
run in `Security Guardrails` as advisory/report-only checks
(`docs/SECURITY_GUARDRAILS.md` lines 13, 45-47, 163-164). They upload SARIF or
soft-fail and never block a merge, so dependency, IaC, and SAST debt can land
silently and `main`'s security posture is documentation, not enforcement.

Root cause: ratcheting a scanner from advisory to blocking is not a single
switch. Each scanner has (a) an unmeasured-or-unburned finding backlog that
would turn every PR (or `main`) permanently red the moment it blocks, and (b) a
structural gating gap -- every advisory scanner is gated
`if: github.event_name != 'pull_request' && != 'pull_request_target'`
(`.github/workflows/security_guardrails.yml` lines 126, 160, 173, 210, 239), so it runs only on
push-to-`main` and the weekly schedule and produces no pull-request check
context. A scanner that never runs on a PR cannot be added to branch protection
as a required PR check the way #1827 added the Gitleaks contexts.

Because the five scanners have very different backlog sizes and blast radii,
flipping them in an undocumented ad-hoc order risks either a perma-red `main`
or a fake gate (the exact failure modes #1827 fought). This slice ships the
ordered ratchet program as a plan doc so each subsequent advisory-to-blocking
flip is a small, owned, sequenced follow-up with a known precondition, instead
of a one-shot "make everything required" change. No workflow or scanner
behavior changes in this PR.

## Scope (this PR)

Ownership lane: security/scanner-ratchet
Slice phase: Workflow/process

1. Add `plans/PR-Security-Scanner-Ratchet.md` defining the ordered ratchet
   program (slices R1-R5) with a per-scanner precondition, burn-down step, and
   advisory-to-blocking flip, plus the shared PR-context decision that gates
   any scanner becoming a required PR check.
2. Record the measured current state (which scanner is red where, and why) so a
   later slice does not re-derive it, and capture the one concrete blocker bug
   found during investigation (pip-audit cannot resolve `requirements.asr.txt`).
3. Ship no workflow, scanner-config, or product-code change. Each R-slice is a
   separate follow-up PR with its own plan and <400 LOC budget.

### Review Contract

Acceptance criteria:

- The plan enumerates all five advisory scanners (Semgrep, OSV, Trivy, Checkov,
  pip-audit) with an explicit ratchet order and the precondition for each flip.
- The plan names the structural PR-context gap and the decision each R-slice
  must make (push-blocking only vs. add a PR-time diff/baseline-gated required
  context), and states that any new required context must also update
  `scripts/check_required_status_checks.py` and live branch protection.
- The plan records the pip-audit `invalid-egg-fragment` blocker on
  `requirements.asr.txt` and ties it to slice R2.
- This PR changes only `plans/PR-Security-Scanner-Ratchet.md`; no workflow or
  code behavior changes.

Affected surfaces:

- Security guardrail ratchet sequencing (planning only).
- No runtime, CI gate, or scanner-config change in this PR.

Risk areas:

- A roadmap plan that drifts from the live workflow: mitigated by recording
  measured run state and file:line anchors a reviewer can reproduce.
- Over-scoping: this PR intentionally ships only the doc; the actual flips are
  deferred to R1-R5.

Triggered reviewer rules:

- R1 Requirements match
- R8 CI/workflow safety
- R14 Codebase verification

### Files touched

- `plans/PR-Security-Scanner-Ratchet.md`

## Mechanism

### Measured current state (origin/main, latest runs as of 2026-06-25)

`Security Guardrails` on push-to-`main` is currently RED, and the sole failing
job is **OSV** (`OSV dependency scan / osv-scan`); every other job is green
only because it is soft:

- pip-audit: `continue-on-error: true` (`security_guardrails.yml:155`)
- Semgrep SAST: runs without `--error`, upload-only (`:193`)
- Trivy config: `exit-code: "0"`, HIGH/CRITICAL only (`:226`)
- Checkov: `soft_fail: true` (`:259`)
- OSV: reusable `google/osv-scanner-action`, no soft-fail -> fails on findings.

The nightly no-baseline `Security Full Sweep` is RED by design (its own
tripwire): pip-audit (SCA), Semgrep (whole-tree), and Gitleaks (full-history)
all fail. That workflow stays red until buried debt is triaged and is NOT a
target to silence here.

Backlog counts for Semgrep / OSV / Trivy / Checkov / Gitleaks live in the
GitHub Security -> code-scanning tab (each uploads SARIF). pip-audit's true
count is currently unmeasurable -- see R2.

### Shared prerequisite: the PR-context decision

Every advisory scanner only runs on push/schedule, so none can be a required PR
check today. Each R-slice must choose one:

- (a) **Push-blocking only** -- remove the soft-fail so the job blocks
  push-to-`main` (and the weekly schedule). Gates `main`, keeps PRs fast, but
  catches debt only after merge.
- (b) **PR-time diff/baseline-gated** -- add a `pull_request` job that scans
  only the diff or compares against a baseline (the Gitleaks model #1827
  established), producing a required check context. Gates PRs before merge.

When an R-slice takes path (b) and wires a new required context, it MUST also
extend `scripts/check_required_status_checks.py` (required-context set + app-id
pin) and the live `main` branch-protection setting, exactly as #1827 did, or
the audit will (correctly) report drift. Recommended: (b) with diff-gating for
the large-backlog scanners (OSV, Semgrep); (a) first then (b) for the
small-backlog scanners (Trivy, Checkov, pip-audit).

### Ratchet order (ascending backlog / blast radius)

**R1 -- Trivy + Checkov to blocking (lowest backlog).**
Both are already green on push in soft mode, so the backlog is plausibly near
zero. Precondition: confirm zero (or triage) HIGH/CRITICAL in the latest SARIF.
Flip: Trivy `exit-code: "1"`, Checkov `soft_fail: false`. Start on path (a);
optionally add path (b) once green. Target files:
`.github/workflows/security_guardrails.yml`, `docs/SECURITY_GUARDRAILS.md`,
`HARDENING.md`.

**R2 -- pip-audit resolvable, measured, then blocking.**
Blocker bug (regression from `PR-ASR-Requirements-Audit-Pin`): `requirements.asr.txt`
line 7 pins NeMo as
`nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@<sha>#egg=nemo_toolkit[asr]`.
pip-audit rejects the trailing `#egg=nemo_toolkit[asr]` fragment
(`invalid-egg-fragment`) and aborts before auditing anything, so the file is in
the matrix but unaudited; `continue-on-error` hides this on `main`. Fix: drop
the redundant `#egg=...` fragment (the `name[extra] @ URL` prefix already
declares the extras). Then measure the true vuln set across all seven matrix
requirement files, triage/upgrade/pin or add justified ignores, and only then
remove `continue-on-error`. Target files: `requirements.asr.txt`,
`.github/workflows/security_guardrails.yml`, `HARDENING.md`,
`docs/SECURITY_GUARDRAILS.md`.

**R3 -- OSV to gate-ready.**
OSV is already hard (no soft-fail) and already the sole red job on push, so the
backlog is real and measurable in code-scanning. Burn down the flagged
vulnerabilities (upgrade/pin) or add an OSV ignore config (osv-scanner.toml)
with justification and review-by dates. Once green it already blocks push; for
PR gating add a diff-aware OSV job (path b). Target files:
`.github/workflows/security_guardrails.yml`, a new OSV ignore config, and
`docs/SECURITY_GUARDRAILS.md`.

**R4 -- Semgrep diff-gated blocking.**
The whole-tree SAST backlog is the largest; clearing all historical findings is
impractical and not required for value. Adopt diff-aware scanning
(`--baseline-commit` against the merge-base) so only NEW findings block, and
keep the whole-tree scan in the nightly full sweep as the tripwire. Path (b) by
construction. Target files: `.github/workflows/security_guardrails.yml` (and/or
a new PR-time semgrep workflow), `.semgrepignore`/config,
`docs/SECURITY_GUARDRAILS.md`.

**R5 -- Gitleaks full-history to blocking (tied to deferred #1).**
Not a switch flip: the full-history scan is red because of real historical
secrets, whose correct remediation is credential rotation + history scrub
(#1827 deferred item #1: commit `d63a9b77...`, plus the archived IndexNow key),
not a gate change. Sequenced last and cross-referenced to that rotation work.
The PR-time Gitleaks scan is already required (#1827), so this only concerns the
full-history tripwire.

## Intentional

- This PR is plan-only (plan-first per AGENTS.md). It ships the ordered program
  so each flip is a small owned slice, not one risky "require everything" change.
- Order is ascending backlog/blast radius (Trivy/Checkov -> pip-audit -> OSV ->
  Semgrep -> Gitleaks-history) so the cheapest, safest wins land first and each
  later slice inherits a proven pattern.
- The nightly `Security Full Sweep` is intentionally left red-by-design; this
  program does not try to silence the no-baseline tripwire.
- The PR-context decision is surfaced as a per-slice choice rather than fixed
  here, because the right answer differs by backlog size and the doc
  (`SECURITY_GUARDRAILS.md:8`) deliberately keeps the full suite off PRs for
  wall-clock reasons.

## Deferred

- The actual advisory-to-blocking flips R1-R5 -- each ships as its own
  follow-up PR with its own plan and <400 LOC budget.
- Credential rotation / history scrub for commit `d63a9b77...` and the archived
  IndexNow key (#1827 deferred item #1); it gates R5.
- Per-image Trivy image scans on production image publish workflows.
- ZAP baseline rule tuning after the first configured staging run.

Parked hardening: none.

## Verification

- Plan sync check: `scripts/sync_pr_plan.py --check` on this plan doc -- passes.
- ASCII-only Python gate is unaffected (no Python files changed); `scripts/check_ascii_python.sh` -- passes.
- Whitespace diff check: `git diff --check` -- passes.
- No code or workflow change, so no unit/integration tests apply to this PR;
  the R1-R5 slices carry their own verification.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Security-Scanner-Ratchet.md` | 213 |
| **Total** | **213** |

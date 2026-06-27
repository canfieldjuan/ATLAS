# PR-Dependabot-Package-Maintenance-Wave

## Why this slice exists

After the `#1819` npm security bundle landed, the remaining Dependabot package
PRs that touch UI package manifests and lockfiles needed to be rebased and
processed one at a time. The repository PR body contract requires every PR body
to point at an existing `plans/PR-*.md` document, but Dependabot's generated
bodies do not create plan documents and editing Dependabot branches would mix
policy scaffolding into otherwise dependency-only diffs.

This shared plan gives the package-maintenance wave one real plan reference that
can be used by the remaining Dependabot PR bodies while keeping those bot
branches scoped to dependency manifests and lockfiles only.

## Scope (this PR)

Ownership lane: dependency-maintenance
Slice phase: Workflow/process

1. Add a shared plan document for the remaining Dependabot UI/package PRs.
2. Use this document as the plan reference in metadata-only PR body updates for
   Dependabot package PRs when their branches remain dependency-only.
3. Keep dependency changes in their original Dependabot PRs; this slice does not
   update any package versions or lockfiles.

### Review Contract

Acceptance criteria:

- The plan document itself passes the repo's plan-shape and plan-file checks.
- Dependabot package PR bodies can reference this existing plan document without
  adding plan files to bot branches.
- No package manifest, lockfile, source, workflow, or Docker file changes are
  included in this slice.

Affected surfaces:

- PR metadata policy support for the Dependabot package-maintenance queue.

Risk areas:

- Over-broad plan reference: mitigated by using this only for the queued
  Dependabot package PRs whose diffs are verified separately to contain the
  expected package manifest and lockfile changes.

Triggered reviewer rules:

- R2 Test evidence
- R14 Codebase verification

### Files touched

- `plans/PR-Dependabot-Package-Maintenance-Wave.md`

## Mechanism

The plan is intentionally documentation-only. Each Dependabot package PR remains
responsible for its own version bump, changed-file verification, and CI result.
When a Dependabot PR body needs to satisfy the repository contract, its body can
point at this existing plan document and list the exact package, directory,
changed files, and verification state for that PR. This avoids committing plan
files into bot branches while preserving the repository's review ritual.

## Intentional

- Documentation-only slice; no dependency version changes land here.
- Keeps Dependabot branches clean and dependency-only.
- Supports the risk-first attack order after `#1819`: low-blast-radius dev
  dependency bumps first, Tailwind major bumps later.
- No `plans/INDEX.md` update; active plans live in the plans root until the
  normal archive flow runs.

## Deferred

- Merging individual Dependabot package PRs; each remains gated by its own diff
  inspection and required checks.
- Tailwind 3 to 4 visual smoke checks; those belong to the individual Tailwind
  PRs.
- Any policy changes to exempt Dependabot PR bodies from the plan contract.

Parked hardening: none.

## Verification

- Expected CI: `pre-push-audit` should validate this plan shape, file list, diff
  size table, and PR body contract.
- Expected CI: `Security Guardrails`, `AI Reconciliation`, and Vercel should run
  unchanged because this is documentation-only.
- Local verification was not run from this projectless workspace because the repo
  is not checked out locally.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Dependabot-Package-Maintenance-Wave.md` | 78 |
| **Total** | **78** |

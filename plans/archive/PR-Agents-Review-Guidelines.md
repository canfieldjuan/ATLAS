# PR-Agents-Review-Guidelines

## Why this slice exists

The GitHub Codex connector reviews Atlas PRs, but it does not follow Atlas's
review process: it ingests review guidance only from a `## Review guidelines`
section in `AGENTS.md`, and Atlas's `AGENTS.md` has no such section. So the
connector runs its generic default review instead of the Reviewer Rules Pack
(`docs/REVIEWER_RULES.md`) the Claude reviewer session uses. Operator request:
make the Codex connector review Atlas the same way it reviews our other repos.

### Problem-derived contract

- Root cause: The Codex connector reads review guidance only from a `## Review guidelines` section of `AGENTS.md`. Atlas has no such section, so its rule pack (`docs/REVIEWER_RULES.md`) and Review Contract discipline never reach the connector.
- Correct fix must touch/change: `AGENTS.md` — add a `## Review guidelines` section that anchors automated reviewers (the connector) to `docs/REVIEWER_RULES.md` (R1–R14), `file:line` + rule-ID citations, the PR Review Contract, and the BLOCKER/MAJOR/NIT/LGTM verdict taxonomy.
- Must not change: `docs/REVIEWER_RULES.md` (remains the single source of truth for the rules), the reviewer-session / merge-gate scripts, the numbered workflow sections of `AGENTS.md`, and any code, tests, or product behavior.

## Scope (this PR)

Ownership lane: workflow/pr-contract
Slice phase: Workflow/process

1. Add a `## Review guidelines` section to `AGENTS.md` that points automated reviewers (the Codex connector) at `docs/REVIEWER_RULES.md` and the PR Review Contract.
2. No new code or tests; the connector reads the section at review time (proof is the connector applying the rules on subsequent PRs).

### Files touched

- `AGENTS.md`
- `plans/PR-Agents-Review-Guidelines.md`

### Review Contract

- Acceptance criteria:
  - [ ] The section uses the exact `## Review guidelines` header the connector reads.
  - [ ] It references `docs/REVIEWER_RULES.md` (R1–R14) and the PR `### Review Contract`.
  - [ ] Verdict taxonomy matches the pack (BLOCKER / MAJOR / NIT / LGTM) and requires `file:line` + rule-ID citations.
  - [ ] No other `AGENTS.md` content or any code/tests changed.
- Reachability proof: N/A (docs-only; the connector reads the section during review).
- Affected surfaces: documentation (`AGENTS.md`).
- Risk areas: none (docs-only; no runtime behavior).
- Reviewer rules triggered: R1, R14.

## Mechanism

The new `## Review guidelines` section sits directly after the two-session intro
in `AGENTS.md`. The Codex connector searches the repo for a `## Review guidelines`
section and follows it; anchoring that section to `docs/REVIEWER_RULES.md` makes
the connector apply Atlas's own R1–R14 rules, cite rule IDs + `file:line`, honor
the PR Review Contract, and use the BLOCKER/MAJOR/NIT/LGTM taxonomy — the same
discipline the Claude reviewer session already follows. The rule pack stays the
single source of truth; the section is a thin entry point to it.

## Intentional

- The section restates a few essentials (cite rule ID + `file:line`, verdict taxonomy, verify-against-codebase) inline rather than only linking `docs/REVIEWER_RULES.md`, so the connector has the discipline in-context on every review. The rule pack remains authoritative for the full R1–R14 detail; the inline restatement is a deliberate, small redundancy for reviewer convenience.

## Deferred

- None. The sibling repos atlas-portfolio and atlas-memory get the equivalent hook in their own repositories' PRs.

Parked hardening: none.

## Verification

- bash scripts/local_pr_review.sh — passed.
- Manual: `AGENTS.md` renders without Markdown errors; the cross-reference to `docs/REVIEWER_RULES.md` resolves.

## Estimated diff size

| File | LOC |
|---|---:|
| `AGENTS.md` | 23 |
| `plans/PR-Agents-Review-Guidelines.md` | 71 |
| **Total** | **94** |

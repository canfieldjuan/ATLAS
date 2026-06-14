# PR-Spark-Subagent-Routing-Docs

## Why this slice exists

The operator wants lightweight Spark subagents used automatically for bounded
read-only scouting when that can reduce main-session context usage. The current
workflow already says lookup belongs in subagents, but it predates the Spark
preference and still names the older Explore/Kimi shapes. This slice codifies
the new default without changing the core safety boundary: main owns judgment,
edits, Git/GitHub mutations, and final synthesis.

## Scope (this PR)

Ownership lane: workflow/process
Slice phase: Workflow/process

1. Update the within-session agent routing contract so Spark is the preferred
   lightweight read-only scout for narrow checks when available.
2. Add a fresh-session bootstrap reminder so restarted sessions apply the same
   Spark default without the operator having to ask.

### Files touched

- `AGENTS.md`
- `docs/SESSION_BOOTSTRAP.md`
- `plans/PR-Spark-Subagent-Routing-Docs.md`

### Review Contract

Acceptance criteria:

- The docs distinguish Spark read-only scouting from main-session judgment and
  mutation work.
- The docs preserve the existing boundaries: main reads files it will edit,
  `rg`/Bash owns exact lookups, main owns review verdicts and final answers.
- Fresh-session bootstrap includes a concise reminder to use Spark for bounded
  read-only scouting.

Affected surfaces:

- Atlas builder/reviewer workflow documentation only.

Risk areas:

- Over-broad wording could imply Spark may make merge/review/code decisions.
- Naming Spark too narrowly could fail to preserve Explore as the wider
  orientation fallback.

Reviewer rules triggered: R1, R12, R14

## Mechanism

`AGENTS.md` section 5 gets an explicit Spark row in the routing table and a
short subsection that defines Spark as the preferred lightweight read-only scout.
The existing worker-model subsection remains as the cheaper single-file fallback
language. `docs/SESSION_BOOTSTRAP.md` gets a one-bullet reminder under context
discipline so fresh sessions inherit the same policy.

## Intentional

- No code or tool wrapper changes. The goal is to codify routing policy, not to
  hardwire model selection in scripts.
- Spark is framed as "if available" so sessions without that subagent surface
  can fall back to Explore or direct reads.
- Judgment, GitHub mutations, exact edit-target reads, and final synthesis stay
  in main.

## Deferred

- No automated enforcement for subagent choice. The routing decision remains an
  operating-model rule, not a mechanical gate.

Parked hardening: none.

## Verification

- Doc diff inspection: passed.
- `python scripts/sync_pr_plan.py plans/PR-Spark-Subagent-Routing-Docs.md --check`: passed.
- Body-aware local PR review: passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `AGENTS.md` | 18 |
| `docs/SESSION_BOOTSTRAP.md` | 1 |
| `plans/PR-Spark-Subagent-Routing-Docs.md` | 88 |
| **Total** | **107** |

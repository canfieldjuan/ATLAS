# PR-Reconstruction-Protocol

## Why this slice exists

Reviews that trust the PR description reproduce the author's framing and miss
where the diff diverges from a correct fix. This codifies the independent
reconstruction method as the mandatory review protocol so it applies to every
reviewer -- human Claude sessions and the Codex/reconciliation gate -- and
survives compactions and new sessions. Exemplar: on #1999 a review posted a
BLOCKER + "six blocking findings" read off the description and stale bot
thread-titles that the head code had already resolved; reconstructing from the
diff caught it.

## Scope (this PR)

Ownership lane: workflow/process
Slice phase: Workflow/process

1. Add `docs/PR_RECONSTRUCTION_PROTOCOL.md` -- the canonical 4-step protocol
   (read the diff alone; derive the correct fix from the problem alone; report
   every gap; cite file:line, sort confirmed/contradicted/could-not-determine,
   gaps first).
2. Make it mandatory in the reviewer workflow by referencing it at the head of
   `AGENTS.md` §4a (Independent verification).

### Review Contract

- Acceptance criteria:
  - [ ] `docs/PR_RECONSTRUCTION_PROTOCOL.md` states the 4 ordered steps and the
        confirmed/contradicted/could-not-determine sorting with file:line
        citations and gaps-first ordering.
  - [ ] `AGENTS.md` §4a names the protocol as mandatory for every review and
        links the doc.
  - [ ] Docs-only change; no code, config, migration, or test touched.
- Reachability proof: N/A -- docs/process-only change with no runtime, UI,
  report, billing, or public-contract surface. Proof is the rendered doc + the
  AGENTS.md reference.
- Affected surfaces: the reviewer contract (`AGENTS.md`) and a new docs file.
- Risk areas: none (documentation); the protocol composes with, and does not
  replace, `docs/REVIEWER_RULES.md` (R1-R14) or §4a's existing challenger pass.
- Reviewer rules triggered: R1.

### Files touched

- `AGENTS.md`
- `docs/PR_RECONSTRUCTION_PROTOCOL.md`
- `plans/PR-Reconstruction-Protocol.md`

## Mechanism

A new markdown doc plus a paragraph inserted at the top of `AGENTS.md` §4a that
names the protocol mandatory and links the doc. No code paths change.

## Intentional

- Codified in the repo (not only in a session memory) so it reaches Codex and
  the reconciliation gate, not just one session lineage. (No automated CI review
  action exists in the repo today; wiring one is a when-added follow-up.)
  Session-local layers (operator memory, global reviewer rules) mirror it for
  redundancy but the repo doc is the canonical source.

## Deferred

- None.

Parked hardening: none.

## Verification

- `scripts/audit_plan_doc.py` on this plan -- OK.
- `git diff --name-status` -- one new doc + `AGENTS.md` + this plan; no code.

## Estimated diff size

| File | LOC |
|---|---:|
| `AGENTS.md` | 28 |
| `docs/PR_RECONSTRUCTION_PROTOCOL.md` | 49 |
| `plans/PR-Reconstruction-Protocol.md` | 80 |
| **Total** | **157** |

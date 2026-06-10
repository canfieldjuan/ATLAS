# REVIEW_MISSES.md - the reviewer-side flywheel

> Mirror of `HARDENING.md`, pointed at the **reviewer** instead of the builder.
> The recurring-lapse list in `docs/SESSION_BOOTSTRAP.md` already turns every
> *builder* mistake into a durable gate. This ledger does the same for every
> **escaped defect**: a bug that shipped past an approved review.

## The rule

**No escaped defect is fixed only once. It must become a gate.**

When a defect is found *after* a PR was approved and merged, log it here, then
convert it into exactly one durable form so it can never silently recur:

1. A mechanical audit (`scripts/audit_*.py` / `check_*.py`) - preferred. Ships
   with fixture tests proving its failure branch fires (`AGENTS.md` 3h/3i).
2. A new rule ID in `docs/REVIEWER_RULES.md`.
3. A new path-based rule trigger in `docs/REVIEWER_RULES.md`.
4. A line in the recurring-lapse checklist (`docs/SESSION_BOOTSTRAP.md`).
5. A change to the Review Contract template (the Scope-section block).

If a miss cannot be converted to a gate, that fact is itself recorded in the
row (column "New gate added" = "judgment-only, see runbook ...") so the residual
human-discipline floor stays visible rather than pretended away.

## How patterns surface

Tally the "Missed by" column over time. The signal to watch, per the operating
model's gap (b): **AI findings the human reviewer missed.** If Codex/Copilot
repeatedly catch a category a human reviewer waves through, that category needs
either a sharper rule, a narrower review scope, or a bootstrap-prompt addition -
the same way a builder's repeated lapse gets front-loaded into the bootstrap.

## Ledger

| Date | Escaped issue | Missed by (human / AI / CI) | Root cause | New gate added | Owner |
|---|---|---|---|---|---|
| _seed_ | _First real entry goes here. Until then this row documents the format._ | - | - | - | - |
| 2026-06-09 | Repeated "fixed the cited example, not the class" pattern across blog-prose quality and raw-ticket clustering recall review rounds. | human/process | Review comments supplied concrete examples without making hardcoding unable to pass; builder did not self-probe with unseen same-class cases before claiming done. | R13 in `docs/REVIEWER_RULES.md` + AGENTS/bootstrap self-probe requirement for 5-10 unseen same-class cases. | reviewer + builder |
| 2026-06-10 | Review conclusions accepted PR/story evidence instead of checked-out codebase evidence: a deep-dive issue set was pinned to a reviewer branch instead of `main`, and one repro used a transcribed function copy instead of the repository code. | human/process | Reviewer source of truth was implicit, so branch/head/codebase verification could be skipped without the verdict naming what was and was not verified. | Promoted -> R14 in `docs/REVIEWER_RULES.md` + AGENTS reviewer template/checklist/bootstrap requiring reviewed head, code/caller/test/artifact spot-checks, and "not verified" disclosure. | reviewer |

## Lifecycle (so this stays a queue, not an archive)

Like `HARDENING.md`, this is a working queue. A row is "open" until its gate
lands; once the gate is merged, mark the row resolved (or move resolved rows to
a dated section). This file must inherit the same retirement discipline tracked
in issue #1319 so it does not become write-only sediment.

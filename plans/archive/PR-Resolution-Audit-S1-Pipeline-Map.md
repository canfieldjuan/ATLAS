# PR-Resolution-Audit-S1-Pipeline-Map

## Why this slice exists

Slice 1 of the Resolution Audit CSV audit arc (epic #1956, slice issue #1957).
Before judging the pipeline (raw CSV upload -> Snapshot -> Full Report), the arc
needs a verified, `file:line`-anchored map of the actual data flow. This pipeline
has a specific trap: the file named `support_ticket_clustering.py` is not where
the customer-facing `ticket_count` (and every `$13.50 x count` dollar figure) is
computed. This slice establishes the shared ground truth so Slices 2-4 attack the
right stages.

## Scope (this PR)

Ownership lane: workflow/process
Slice phase: workflow/process

1. Map the six Phase-0 areas: ingestion, normalization, clustering, metrics/cost,
   snapshot-vs-full-report, verification/scorecard.
2. Inventory dead code / orphaned modules.
3. Record a table of flags carried into later slices.
4. No product code changes; gitignore `_audit_scratch/` for later empirical work.

### Review Contract

- Acceptance criteria:
  - `docs/audits/resolution-audit-csv/PIPELINE_MAP.md` maps the six Phase-0 areas
    with `file:line` anchors verified against `4f2790e6d`, and every claim is
    accurate against the code at that commit (corrected after the first Codex
    review).
  - No product code is modified; `_audit_scratch/` is gitignored.
- Affected surfaces: documentation only. No runtime, API, auth, billing,
  delivery, webhook, or UI surface is introduced, so no reachability proof applies.
- Rule mapping: the diff triggers no review-rule classes (no auth/token/permission,
  no webhooks/jobs, no defect-class review comments) -- it is a read-only audit map.

### Files touched

- `.gitignore`
- `docs/audits/resolution-audit-csv/PIPELINE_MAP.md`
- `plans/PR-Resolution-Audit-S1-Pipeline-Map.md`

## Mechanism

Three independent read-only mapping passes (ingestion+normalization,
clustering+metrics, snapshot+scorecard) over the pipeline at `origin/main`
@ `4f2790e6d`, each producing `file:line` anchors, synthesized into one map at
`docs/audits/resolution-audit-csv/PIPELINE_MAP.md`. No fixtures or code execution
in this slice -- that is Slice 2 (#1958).

## Intentional

- Structure/map only; severity-rated defects are deferred to `FINDINGS.md`.
- Captures the two-clustering-systems orientation and that `ticket_faq_markdown.py`
  (not `support_ticket_clustering.py`) computes the billed counts.
- Product code untouched; experiments confined to gitignored `_audit_scratch/`.

## Deferred

- `FINDINGS.md` (severity-rated) -- Slice 2 (#1958, CSV parsing + clustering) and
  Slice 3 (#1959, performance).
- `REFACTORS.md`, `PRESENTATION.md`, one-page `SUMMARY.md` -- Slice 4 (#1960).

## Verification

- Read-only investigation; no product code in this diff, so no unit/integration
  suite applies.
- Every `file:line` anchor produced against `origin/main` @ `4f2790e6d` and
  cross-checked across the three mapping passes.
- `python scripts/audit_pr_body.py --pr-author canfieldjuan <body>` (PASS).
- `git diff --check` clean; ASCII-only markdown.

## Estimated diff size

| File | LOC |
|---|---:|
| `.gitignore` | 3 |
| `docs/audits/resolution-audit-csv/PIPELINE_MAP.md` | 272 |
| `plans/PR-Resolution-Audit-S1-Pipeline-Map.md` | 80 |
| **Total** | **355** |

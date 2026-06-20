# PR-Deflection-Parser-Lane-Closeout

## Why this slice exists

#1675 merged the last active parser/admission hardening PR in this session.
The remaining parser issues (#1467/#1582/#1463) now need a clean closeout
record: which parser cliffs were fixed, where the real observed evidence lives,
and why the low non-zero reject threshold is still blocked instead of being
invented from synthetic fixtures.

Root cause: the issue tracker and `plans/` root were not reconciled after the
parser-testing sequence merged several small PRs in rapid succession. That
leaves merged plans looking in-flight and leaves the threshold-policy boundary
buried in comments instead of in a durable repo artifact.

This fixes the root for this lane by archiving only verified merged parser
plans and adding a sanitized closeout note that points to the observed evidence
without committing raw customer/export data. It does not change parser behavior.

## Scope (this PR)

Ownership lane: content-ops/deflection-parser-testing
Slice phase: Workflow/process

1. Archive this session's verified-merged parser plan docs:
   #1657, #1662, #1667, #1673, and #1675.
2. Refresh `plans/INDEX.md` after those moves.
3. Add a sanitized closeout note for #1467/#1582/#1463 that names the
   completed slices, the observed CSV evidence artifact, and the remaining
   threshold-policy boundary.
4. Do not archive or edit other lanes' root plans.

### Review Contract

Acceptance criteria:
- Every plan moved to `plans/archive/` corresponds to a verified merged parser
  PR in this session.
- The closeout note references only sanitized counts/status, not raw observed
  CSV contents, customer text, request IDs, or secrets.
- The closeout note does not claim #1467's low non-zero reject threshold is
  solved; it keeps the policy blocked on real evidence or an explicit product
  decision.
- `plans/INDEX.md` reflects the archive moves.

Affected surfaces:
- Plan archive/index only.
- Parser validation documentation only.
- GitHub issue comments after the PR opens.

Risk areas:
- Accidentally archiving concurrent in-flight plans from another lane.
- Overstating parser launch readiness by closing the threshold policy without
  the real evidence #1467 asked for.
- Leaking raw observed CSV content in a committed artifact.

- Reviewer rules triggered: R1, R2, R3, R12, R14.

### Files touched

- `docs/extraction/validation/deflection_parser_lane_closeout_2026-06-17.md`
- `plans/INDEX.md`
- `plans/PR-Deflection-Parser-Lane-Closeout.md`
- `plans/archive/PR-Deflection-Parser-CSV-Field-Limit.md`
- `plans/archive/PR-Deflection-Parser-Diagnostics-Parse-Errors.md`
- `plans/archive/PR-Deflection-Parser-JSON-Message-Guard.md`
- `plans/archive/PR-Deflection-Parser-JSONL-Line-Diagnostics.md`
- `plans/archive/PR-Deflection-Parser-Observed-Text-Aliases.md`

## Mechanism

The plan archive uses explicit `git mv` operations for only the five parser
plans whose PRs were verified as merged through `gh pr view`. The archive index
is regenerated with `python scripts/archive_plans.py index`, matching the
existing archive workflow.

The closeout note summarizes the parser-testing run in one sanitized document:
completed PRs, the already-committed observed-evidence artifact, and the
remaining threshold boundary. It keeps the raw observed CSV files out of git and
states that the only real partial case found so far is explained by
private/internal row filtering, so it should stay ACCEPT-with-warning rather
than become a hard reject.

## Intentional

- No parser code changes: the active parser hardening work is merged, and a
  new threshold policy would be speculative without a real low-coverage
  provider export.
- No bulk archive sweep: this PR only moves the verified merged parser plans
  owned by this session.
- No raw observed CSV attachment: the committed artifact stays at sanitized
  counts and interpretation.

## Deferred

- #1467 low non-zero reject threshold remains blocked until real evidence
  shows parser uncertainty rather than intentional private/internal filtering,
  or until the operator makes an explicit product decision to warn/reject more
  aggressively.

Parked hardening: none.

## Verification

- Verified PR merge state for #1657, #1662, #1667, #1673, and #1675 with
  `gh pr view`.
- Synced and checked the plan with `scripts/sync_pr_plan.py`.
- Ran this redaction grep over the closeout note and plan; it returned no
  matches:

  ```bash
  rg -n -i '(/(h)ome/|https://[A-Za-z0-9./?=_&%-]+/systems/support-ticket-deflection/results/[A-Za-z0-9_-]+|request[_-]?id[:=][[:space:]]*[[:alnum:]_-]+|cs_(live|test)_[[:alnum:]_]+|pi_[[:alnum:]_]+|ch_[[:alnum:]_]+|sk_(live|test)_[[:alnum:]_]+|rk_(live|test)_[[:alnum:]_]+|whsec_[[:alnum:]_]+|Bearer[[:space:]]+[[:alnum:]._-]+|[[:alnum:]._%+-]+@[[:alnum:].-]+[.][[:alpha:]]{2,})' docs/extraction/validation/deflection_parser_lane_closeout_2026-06-17.md plans/PR-Deflection-Parser-Lane-Closeout.md
  ```
- Manually inspected the closeout note for quoted/fenced raw observed CSV rows
  or customer-message samples; none are present.
- `scripts/local_pr_review.sh` with the PR body file: passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/extraction/validation/deflection_parser_lane_closeout_2026-06-17.md` | 56 |
| `plans/INDEX.md` | 7 |
| `plans/PR-Deflection-Parser-Lane-Closeout.md` | 129 |
| `plans/archive/PR-Deflection-Parser-CSV-Field-Limit.md` | 0 |
| `plans/archive/PR-Deflection-Parser-Diagnostics-Parse-Errors.md` | 0 |
| `plans/archive/PR-Deflection-Parser-JSON-Message-Guard.md` | 0 |
| `plans/archive/PR-Deflection-Parser-JSONL-Line-Diagnostics.md` | 0 |
| `plans/archive/PR-Deflection-Parser-Observed-Text-Aliases.md` | 0 |
| **Total** | **192** |

# PR-Reviewer-Path-Trigger-Audit

## Why this slice exists

S1 (PR #1330) added the path-to-rule trigger table to `docs/REVIEWER_RULES.md`
as reviewer discipline, and named the mechanical enforcer as deferred. This
slice S3 of the review-workflow redesign (issue #1328) builds it: an audit that
derives the rule IDs a PR's changed files trigger and fails when the plan's
Review Contract "Reviewer rules triggered" line omits one. It closes the loop so
a checker PR cannot silently ship without declaring R2/R10, etc.

Diff budget: ~530 LOC, over the 400 soft cap. The plan doc is ~116; the rest is
the audit (250) plus its fixtures (156), grown by review-round hardening
(wrapped-line declarations, mixed-row prose surfacing). The detector and its
failure-proving coverage are the irreducible substance and do not split cleanly.

## Scope (this PR)

Ownership lane: dev-workflow/review-contract
Slice phase: Workflow/process

1. Add `scripts/audit_review_rules_triggered.py`: parse the trigger table from
   `docs/REVIEWER_RULES.md`, map the diff's changed files to required rule IDs,
   and fail when the plan omits a triggered rule. Prose-only trigger rows are
   surfaced as advisory, never silently skipped.
2. Add `tests/test_audit_review_rules_triggered.py`: fixtures for table parsing,
   glob matching, required-vs-declared, the missing-rule detection branch, the
   prose-row surfacing, and the CLI usage-error exit code.
3. Wire the audit into `scripts/local_pr_review.sh` (per committed plan doc,
   reusing the existing loop) and enroll the test in
   `.github/workflows/pre_push_audit.yml`.

### Review Contract

- Acceptance criteria:
  - [ ] The audit fails (exit 1) when the diff triggers a rule the plan's
        "Reviewer rules triggered" line omits, naming the triggering path.
  - [ ] It passes when the plan declares every triggered rule, and is a no-op
        for a plan-less PR (out of scope, gap (c)).
  - [ ] Prose-only trigger rows are reported as advisory, not dropped.
  - [ ] The trigger table in `docs/REVIEWER_RULES.md` stays the single source of
        truth (parsed, not duplicated in the script).
- Affected surfaces: dev workflow tooling only (one new script, one new test,
  one bundle edit, one workflow edit). No product surface.
- Risk areas: a false-positive blocking a compliant PR (a glob over-matching) or
  a false-negative (glob under-matching). Mitigated by the glob-matching and
  required-rules fixtures; the table is advisory-by-design for prose rows.
- Reviewer rules triggered: R2 (the audit is a detector -- failure branch
  proven), R10 (maintainable). This PR's own diff triggers exactly these via the
  scripts/audit_*.py row, so the audit dogfoods on itself.

### Files touched

- `.github/workflows/pre_push_audit.yml`
- `plans/INDEX.md`
- `plans/PR-Reviewer-Path-Trigger-Audit.md`
- `plans/archive/PR-Reviewer-Reconciliation-Audit.md`
- `scripts/audit_review_rules_triggered.py`
- `scripts/local_pr_review.sh`
- `tests/test_audit_review_rules_triggered.py`

## Mechanism

`scripts/audit_review_rules_triggered.py` reads the "Path-based rule triggers"
table from `docs/REVIEWER_RULES.md`, splitting each row into backticked globs
plus its `R\d+` rule IDs; rows with no backticked glob (prose like "invoicing /
billing / payment code") become advisory prose-rows. It computes the diff's
changed files from the merge-base of `HEAD` and the base ref, matches each
against the globs (a small `**`/`*` -> regex translation, with extension globs
matched on the basename), and unions the triggered rule IDs. It then reads the
plan's "Reviewer rules triggered" line and fails when a triggered rule is not
declared, naming the path that triggered it. The bundle runs it once per
each committed plans/PR-*.md plan, reusing the loop that already runs plan/code
consistency; parsing and matching helpers are unit-covered directly.

## Intentional

- The trigger table is parsed from `docs/REVIEWER_RULES.md` rather than
  hardcoded, so there is one source of truth and the doc cannot drift from the
  gate (the anti-pattern issue #1318 warns about).
- Prose-only rows (no machine glob) are surfaced as advisory rather than
  enforced or silently dropped, per AGENTS.md section 3g -- the audit cannot
  match "invoicing / billing / payment code" mechanically, and says so.
- A plan-less PR is a no-op (the trigger check needs a declared-rules line);
  enforcing it on plan-less PRs is the separate gap (c) work, left out here.

## Deferred

- Enforcing triggers on plan-less PRs (operating-model gap (c)).
- Promoting high-value prose rows (auth, billing) to real globs so they are
  enforced rather than advisory.
- S4: a reviewer-metrics summarizer over `REVIEW_MISSES.md`.

Parked hardening: none.

## Verification

- `tests/test_audit_review_rules_triggered.py` -- fixtures pass (table parse,
  glob match, required-vs-declared, missing-rule detection, prose surfacing,
  CLI usage error).
- `scripts/audit_review_rules_triggered.py` run on this PR -- triggers R2+R10
  via the scripts/audit_*.py row and the plan declares both (exit 0).
- `scripts/check_ascii_python.sh` -- ASCII-clean.
- `scripts/local_pr_review.sh` -- full mechanical bundle green before push.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/pre_push_audit.yml` | 2 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Reviewer-Path-Trigger-Audit.md` | 117 |
| `plans/archive/PR-Reviewer-Reconciliation-Audit.md` | 0 |
| `scripts/audit_review_rules_triggered.py` | 257 |
| `scripts/local_pr_review.sh` | 4 |
| `tests/test_audit_review_rules_triggered.py` | 167 |
| **Total** | **550** |

# PR-Codex-Max-Files-Budget

## Why this slice exists

The PR fix-mode doc layer (merged in #1715: AGENTS.md 3l + baton +
compact instructions) is advisory only. This slice lands the first
**enforcement** piece from issue #1714: a max-files budget on the plan-doc
files-touched audit. A fix loop that strays beyond its declared file count now
fails a cheap, early gate -- the audit already runs in `scripts/pre_push_audit.sh`
before any test or long CI lane -- instead of being caught late or not at all.
This is the Codex-side enforcement (deterministic at push/CI time); the Claude
Code pre-edit-hook track is a separate later slice under #1714.

## Scope (this PR)

Ownership lane: process/agent-guidance
Slice phase: Production hardening
Max files: 4

The files-touched audit gains an optional, opt-in budget. A plan declaring
`Max files: N` in its *Scope* has the budget enforced automatically; a
`--max-files N` flag overrides it for manual fix-session runs. When neither is
set the audit behaves exactly as before, so non-fix PRs are unaffected. This PR
dogfoods the field on its own plan (`Max files: 4`).

### Files touched

- `scripts/audit_plan_doc_files_touched.py`
- `tests/test_audit_plan_doc_files_touched.py`
- `AGENTS.md`
- `plans/PR-Codex-Max-Files-Budget.md`

### Review Contract

Acceptance criteria:

- [ ] A plan with `Max files: N` fails the audit when the diff changes more than
      N files, with `OVER BUDGET` in output and exit 1.
- [ ] `--max-files N` overrides/supplies the budget for manual runs.
- [ ] With no budget declared and no flag, behavior is unchanged (no cap); the
      existing MISSING/EXTRA logic and exit codes are preserved.
- [ ] No change to how `scripts/pre_push_audit.sh` invokes the audit.

Affected surfaces: the plan-doc files-touched auditor (a gate predicate) + its
tests + one AGENTS.md sentence. No application code.

Risk areas: a budget gate that misfires could block legitimate PRs -- mitigated
by making it strictly opt-in (off unless a plan/flag sets it) and counting the
same `actual` diff set the existing EXTRA/MISSING check already uses.

Reviewer rules triggered: R1, R10, R2.

## Mechanism

`declared_max_files()` reads an optional `Max files: N` line from the plan's
*Scope* section only -- so a digit-only mention in other prose or an example
does not arm the gate -- and fails closed (raising `PlanBudgetError`, exit 2) on
a present-but-malformed value so a typo cannot silently disable the budget. This
mirrors how `scripts/audit_plan_doc_diff_size.py` reads its Total from the plan. `main()` moves to argparse -- positional `plan`, optional `base_ref`
(default `origin/main`), and `--max-files` -- which preserves the existing
two-positional call in `scripts/pre_push_audit.sh`. The budget resolves to the
flag if given, else the declared value; when set and the actual changed-file
count exceeds it, the audit prints `OVER BUDGET` and returns exit 1 alongside the
unchanged MISSING/EXTRA reporting. The budget counts every file in
`git diff --name-only base...HEAD`, including the plan doc, so authors set N to
include it.

## Intentional

- Opt-in only: no budget declared and no flag means the cap is off and the audit
  is byte-for-byte the prior behavior, so existing PRs are untouched.
- Budget sourced from the tracked plan doc (not the gitignored baton) so it is
  enforceable at pre-push/CI; the `--max-files` flag covers manual fix-session
  runs.
- No invocation wiring change -- `scripts/pre_push_audit.sh` already passes the
  plan doc, so declaring the field is enough to enforce it.

## Deferred

Tracked in issue #1714 (not this PR):

- Claude Code pre-edit enforcement (`.claude/` PreToolUse deny +
  SessionStart/PreCompact baton re-hydration hooks) and the `/fix-mode` skill.
- `.gitignore` narrowing to make the Claude Code config shareable.

## Parked hardening

None.

## Verification

- Run the files-touched audit tests in `tests/test_audit_plan_doc_files_touched.py`
  -- existing plus new budget cases pass.
- Dogfood: with `Max files: 4` in this plan and exactly four changed files, the
  audit prints the budget line and exits 0; lowering the field to 3 exits 1 with
  `OVER BUDGET` (reverted before commit).
- Run `scripts/audit_plan_code_consistency.py` on this plan -- all path and
  function claims resolve (`declared_max_files` resolves to its def; the
  `Max files: 4` line has no backticks and is not a claim).
- Run `scripts/local_pr_review.sh` -- full local bundle green, including this
  plan passing its own files-touched budget.

## Estimated diff size

| File | LOC |
|---|---:|
| `scripts/audit_plan_doc_files_touched.py` | 74 |
| `tests/test_audit_plan_doc_files_touched.py` | 118 |
| `AGENTS.md` | 3 |
| `plans/PR-Codex-Max-Files-Budget.md` | ~109 |
| **Total** | **~304** |

# PR-Audit-The-Auditors-1

## Why this slice exists

Across two Copilot review rounds on PRs #483 / #484 / #485, the bot
left 23 actionable comments on our newly-shipped audit stack. Tracing
each catch back to "what would have caught this in our stack first"
identified four recurring root causes:

1. **Silent skip of unknown input** (4 catches). Regex / lookup misses
   a valid input but the auditor reports "OK". Includes the
   `ENV_VAR_LINE` digit bug where `[A-Z_]+` rejected `B2B_CHURN`'s
   "2", silently dropping a real port claim and reporting it as
   MISSING-IN-DOC.

2. **Plan-doc claims about code drift from the code** (5 catches).
   The plan doc's *Mechanism* and *Verification* sections describe
   what the scripts do; the descriptions ship without verification
   that they match the actual code (function names, regex literals,
   command outputs).

3. **Encoding / portability** (5 catches). `Path.read_text()` without
   explicit encoding, `startswith("/")` for absolute-path detection
   missing Windows drives + UNC paths.

4. **Shell hygiene** (3 catches). `set -u` alone (not `set -euo
   pipefail`), `[ -x ]` followed by `bash ...` (executable bit
   irrelevant), hardcoded `origin/main` instead of resolving the
   trunk via `refs/remotes/origin/HEAD`.

Underneath all four is one **structural** root cause: we built
mechanical audits for the **codebase** but not for **our audit
scripts**. Each auditor shipped with one happy-path self-test and
zero or one negative tests. That's enough to confirm the audit
runs; it does NOT catch regex bugs that silently drop valid input,
substring false positives, or platform assumptions.

This slice ships the first half of the fix: AGENTS.md principles
plus two new mechanical audits that grep-lint our scripts/ dir for
the recurring anti-patterns. The companion slice
PR-Audit-The-Auditors-2 backfills per-auditor fixture tests under
`tests/test_audit_*.py`.

## Scope (this PR)

1. **AGENTS.md** -- two new subsections under section 3 (Builder
   workflow):

   - **3e. Auditors must surface, never silently skip.** Encodes
     the "unknown input -> DRIFT, not skip" principle with the
     four real examples from this session, plus the anti-pattern
     vs. the right shape in code.

   - **3f. Auditors ship with fixture tests.** Requires every
     `scripts/audit_*.py` to have a sibling `tests/test_audit_*.py`
     that exercises a happy path, at least one negative case
     specific to its parser, and one pathological input.

2. `scripts/audit_script_hygiene.sh` -- bash grep lint of our
   own scripts/ dir. Flags:

   - `scripts/*.sh` without `set -euo pipefail` near the top.
   - `Path(...).read_text(` in `scripts/audit_*.py` without
     `encoding=`.
   - `startswith("/")` (the POSIX-only absolute path check) inside
     any `_validate_path`-shaped function in `scripts/audit_*.py`.
   - `[ -x ... ]` followed by `bash ...` in `scripts/*.sh` (the
     exec-bit-irrelevant case Copilot caught on PR #483).

3. `scripts/audit_plan_code_consistency.py PLAN_PATH` -- verify
   plan doc *Mechanism* / *Verification* claims match shipped code:

   - Path tokens (any string containing `/` plus a file-extension
     suffix) found in Scope / Mechanism / Verification must exist on
     disk under the repo root.
   - Backticked function-call literals (snake_case, >=4 chars,
     followed by parens) found in Mechanism / Verification must
     appear as a "def NAME" or "async def NAME" line in some .py
     file under scripts/ or atlas_brain/.

   Reports missing / extra; exits 1 on any drift.

### Files touched

- `AGENTS.md` (modified)
- `scripts/audit_script_hygiene.sh` (new)
- `scripts/audit_plan_code_consistency.py` (new)
- `plans/PR-Audit-The-Auditors-1.md` (this file, new)

## Mechanism

### `audit_script_hygiene.sh`

```bash
fail=0

# 1. Every scripts/*.sh must `set -euo pipefail` near the top.
for f in scripts/*.sh; do
    if ! head -10 "$f" | grep -qE '^set -[eu]+o\s+pipefail'; then
        echo "FAIL $f: missing 'set -euo pipefail' near the top"
        fail=1
    fi
done

# 2. scripts/audit_*.py must use encoding="utf-8" on every read_text().
for f in scripts/audit_*.py; do
    if grep -nE '\.read_text\(\s*\)' "$f"; then
        echo "FAIL $f: read_text() without encoding=\"utf-8\""
        fail=1
    fi
done

# 3. Inside scripts/audit_*.py, startswith("/") is a POSIX-only
#    absolute-path check; prefer PurePath.is_absolute().
for f in scripts/audit_*.py; do
    if grep -nE 'startswith\("/"\)' "$f"; then
        echo "FAIL $f: startswith(\"/\") is POSIX-only; use Path.is_absolute()"
        fail=1
    fi
done

# 4. scripts/*.sh: [ -x SCRIPT.sh ] guards followed by `bash SCRIPT.sh`
#    are wrong (executable bit is irrelevant when invoking via bash).
for f in scripts/*.sh; do
    if grep -B0 -A2 '\[ -x ' "$f" | grep -q 'bash '; then
        echo "FAIL $f: '[ -x ... ]' guard before 'bash ...' (exec-bit irrelevant)"
        fail=1
    fi
done

exit $fail
```

### `audit_plan_code_consistency.py`

```
parse_claims(plan_text) -> (paths: set[str], functions: set[str])
    Slice the Mechanism and Verification sections.
    For each backticked identifier in those sections:
        - Looks like "<word>/.../<word>.<ext>" -> path claim
        - Looks like "<snake_case>()" -> function-call claim
    Return as two sets.

verify_paths(paths) -> list[str]
    For each path: must exist on disk under REPO_ROOT.

verify_functions(functions) -> list[str]
    Walk all *.py under scripts/ + atlas_brain/.
    Collect every `def <name>` and `async def <name>`.
    For each claimed function: must be in the def set.

Report failures, exit 1 on any.
```

False-positive risk: prose mentions of common function names
(`get()`, `set()`) would be flagged. Mitigation: require the
backticked function name to be at least 4 characters (the same
heuristic the tool-name auditor uses).

## Intentional

- **Bash for hygiene lint, Python for plan/code consistency.** The
  hygiene checks are all grep one-liners; Python's `re` would be
  overhead. The plan/code consistency checker needs to walk the
  AST of every `.py` file under scripts/ + atlas_brain/; bash
  can't do that cleanly.
- **AGENTS.md sub-sections, not new top-level sections.** The
  principles slot under "Builder workflow" (section 3); they
  apply to *how* the builder writes audit scripts, not to the
  PR/reviewer contract.
- **Hygiene lint is opt-in, not enforced via pre_push_audit.sh.**
  Wiring is deferred to `PR-Pre-Merge-Workflow-And-Wrapper`
  (which already has wrapper-wiring as its job) so this slice
  doesn't couple to PR #483's merge cycle.
- **Plan/code consistency checker has a 4-character minimum on
  function names.** Avoids noise on common Python built-ins
  (`get()`, `set()`, `int()`) that appear in plan-doc prose
  without being a contract claim.
- **AGENTS.md examples include the actual Copilot-caught bugs.**
  Concrete > abstract. Reviewers can pattern-match.
- **No pytest fixtures in this slice.** They're the natural
  follow-up (PR-Audit-The-Auditors-2) and the labor (one test
  file per existing auditor, ~50 LOC each) would push this PR
  far over the soft cap.

## Deferred

- **`PR-Audit-The-Auditors-2`** -- one `tests/test_audit_*.py`
  per existing auditor with positive + negative fixtures, then
  wired into `pytest.ini`. The fixture for
  `audit_mcp_port_assignments.py` must include
  `ATLAS_MCP_B2B_CHURN_PORT=8062` so the digit-in-name regex
  bug can never recur.
- **`PR-Pre-Merge-Workflow-And-Wrapper`** (already deferred from
  earlier slices) will wire `audit_script_hygiene.sh` and
  `audit_plan_code_consistency.py` into the pre-push wrapper
  alongside the existing audits.
- **Backticked regex literals in plan docs verified against the
  cited script.** Higher-precision plan/code consistency check.
  Held until we have a clearer rule for which script a given
  regex belongs to.
- **CI integration (GitHub Actions).** Phase 2 of the pre-merge
  gate plan.

## Verification

Local commands the builder ran (reviewer should reproduce):

```bash
# 1. audit_script_hygiene.sh on this branch -- expected to pass
#    after the fixes from earlier rounds landed on the three
#    open audit PRs (encoding=utf-8 everywhere; set -euo pipefail
#    in pre_push_audit.sh; Path.is_absolute() in manifest auditor;
#    [ -x ... ] -> [ -f ... ] in pre_push_audit.sh).
#    On main TODAY the audits ship via the three open PRs, so on
#    a checkout off main with none of the audit PRs merged, this
#    new hygiene check will pass trivially (no scripts/audit_*.py
#    exists yet).
bash scripts/audit_script_hygiene.sh
echo "exit: $?"

# 2. audit_plan_code_consistency.py on this slice's own plan doc.
#    Expected: every backticked path under "scripts/" exists
#    (they will once committed); every backticked function-call
#    literal resolves to a def in some .py file.
python scripts/audit_plan_code_consistency.py \
    plans/PR-Audit-The-Auditors-1.md
echo "exit: $?"

# 3. Known-bad case: a plan doc that claims a script function
#    that doesn't exist.
echo "## Mechanism
\`\`\`
foo_that_does_not_exist()
\`\`\`" > /tmp/bad-plan.md
python scripts/audit_plan_code_consistency.py /tmp/bad-plan.md
echo "exit: $?"  # expect exit 1 with the missing function flagged.
```

## Estimated diff size

| File | LOC (est) |
|---|---|
| `AGENTS.md` (modified) | ~+90 / -0 |
| `scripts/audit_script_hygiene.sh` | ~75 |
| `scripts/audit_plan_code_consistency.py` | ~140 |
| `plans/PR-Audit-The-Auditors-1.md` | ~225 |
| **Total** | **~530** |

130 LOC over the 400 soft cap. Slice is indivisible: plan + scripts
+ AGENTS.md principle ship together. Splitting AGENTS.md into its
own PR would force a 90-LOC-doc-only PR; splitting the two scripts
each into their own PR would require two ~180-LOC plan docs, net
worse. Same shape as PRs #483, #484, #485 (plan doc is ~42% of the
total).

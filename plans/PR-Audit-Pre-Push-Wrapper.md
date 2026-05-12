# PR-Audit-Pre-Push-Wrapper

## Why this slice exists

The individual audit scripts now exist on `main`: CLAUDE.md MCP counts,
MCP ports, plan shape, files touched, and diff size. The remaining useful
piece from oversized PR #483 is a single command that runs those checks
before a builder opens or updates a PR.

## Scope (this PR)

Add `scripts/pre_push_audit.sh`, add a focused wrapper integration test,
and refresh the coordination row.

### Files touched

- `scripts/pre_push_audit.sh`
- `tests/test_pre_push_audit.py`
- `plans/PR-Audit-Pre-Push-Wrapper.md`
- `docs/extraction/coordination/inflight.md`

## Mechanism

The wrapper resolves trunk via `origin/HEAD` or `origin/main`, runs the
repository-level MCP count and MCP port auditors, finds touched
`plans/PR-*.md` files from branch diff plus working tree state, and runs
shape/files-touched/diff-size checks for each plan.

It also runs `scripts/check_ascii_python.sh` when present. Failures are
accumulated so one run shows every broken mechanical check.

## Intentional

- No git hook installation. The command is manual for now.
- No CI wiring in this slice.
- The wrapper only calls auditors that already landed on `main`.

## Deferred

Git hook installer, GitHub Actions integration, extracted-package
validator orchestration, and broader PR-claim auditing.

## Verification

```bash
python -m pytest tests/test_pre_push_audit.py
bash scripts/pre_push_audit.sh
python scripts/audit_plan_doc.py plans/PR-Audit-Pre-Push-Wrapper.md
python scripts/audit_plan_doc_files_touched.py plans/PR-Audit-Pre-Push-Wrapper.md origin/main
python scripts/audit_plan_doc_diff_size.py plans/PR-Audit-Pre-Push-Wrapper.md origin/main
python -m py_compile tests/test_pre_push_audit.py
git diff --check
```

## Estimated diff size

| File | LOC churn (approx) |
|---|---:|
| `scripts/pre_push_audit.sh` | 89 |
| `tests/test_pre_push_audit.py` | 195 |
| `plans/PR-Audit-Pre-Push-Wrapper.md` | 63 |
| `docs/extraction/coordination/inflight.md` | 4 |
| **Total** | **~351** |

# PR-Audit-Agents-Auditor-Principles

## Why this slice exists

Oversized PR #486 bundled AGENTS.md policy plus two meta-audit scripts
and was rejected as unreviewable. This split lands only the policy:
mechanical auditors must not silently skip unknown input, and new audit
scripts need fixture tests that pin parser behavior.

## Scope (this PR)

1. Add AGENTS.md section 3e, "Auditors must surface, never silently skip."
2. Add AGENTS.md section 3f, "Auditors ship with fixture tests."
3. Refresh the in-flight coordination row for this split.

### Files touched

- `AGENTS.md`
- `plans/PR-Audit-Agents-Auditor-Principles.md`
- `docs/extraction/coordination/inflight.md`

## Mechanism

This is a policy/documentation slice. The new AGENTS.md sections encode
the review lessons from the audit-kit PRs:

- unknown parser input should become `DRIFT` or `UNKNOWN`, not vanish
- safe skips need an inline false-positive rationale
- every new `scripts/audit_*.py` should land with fixture tests covering
  happy path, parser-specific negative cases, and pathological rejects

## Intentional

- No code changes in this slice. The script hygiene and plan/code
  consistency auditors from #486 will be split separately.
- The guidance says "should" for fixture tests rather than retroactively
  failing every legacy audit script in the repo.

## Deferred

- Split `scripts/audit_script_hygiene.sh` from #486.
- Split `scripts/audit_plan_code_consistency.py` from #486.
- Close or replace oversized #486 after useful pieces are harvested.

## Verification

```bash
python scripts/audit_plan_doc.py plans/PR-Audit-Agents-Auditor-Principles.md
python scripts/audit_plan_doc_files_touched.py plans/PR-Audit-Agents-Auditor-Principles.md origin/main
python scripts/audit_plan_doc_diff_size.py plans/PR-Audit-Agents-Auditor-Principles.md origin/main
git diff --check
```

## Estimated diff size

| File | LOC churn (approx) |
|---|---:|
| `AGENTS.md` | 53 |
| `plans/PR-Audit-Agents-Auditor-Principles.md` | 61 |
| `docs/extraction/coordination/inflight.md` | 4 |
| **Total** | **~118** |

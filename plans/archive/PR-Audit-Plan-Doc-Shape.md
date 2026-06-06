# PR-Audit-Plan-Doc-Shape

## Why this slice exists

The oversized audit-kit PR #483 bundled multiple auditors and was
rejected under the repo's review-size gate. This split lands only the
plan-doc shape auditor: a deterministic check that a PR plan contains
the required AGENTS.md sections in order.

This catches plan files that omit the contract sections reviewers rely
on, or that accidentally satisfy `Scope` with a heading like `Out of
scope`.

## Scope (this PR)

1. Add `scripts/audit_plan_doc.py`.
2. Add focused tests for happy path, missing section, out-of-order
   sections, duplicate sections, and CLI boundary behavior.
3. Refresh the coordination row for this split slice.

### Files touched

- `scripts/audit_plan_doc.py`
- `tests/test_audit_plan_doc.py`
- `plans/PR-Audit-Plan-Doc-Shape.md`
- `docs/extraction/coordination/inflight.md`

## Mechanism

`audit_plan_doc.py` reads a plan document, extracts `##` headings, and
checks the seven required sections in AGENTS.md order: Why, Scope,
Mechanism, Intentional, Deferred, Verification, and Estimated diff size.

Section matching uses normalized exact-title allowlists, not substring
matching. That means `## Out of scope` does not satisfy `Scope`.
Duplicate required sections and out-of-order sections are explicit
drift states.

## Intentional

- No pre-push wrapper in this PR. The wrapper comes after individual
  auditors land.
- The auditor accepts `## Scope` and `## Scope (this PR)` because both
  forms are common in current plans.
- The auditor checks only top-level `##` plan sections. Subsections are
  intentionally ignored.

## Deferred

- Files-touched-vs-diff auditing.
- Diff-size estimate auditing.
- Pre-push wrapper integration.
- CI wiring.

## Verification

```bash
python -m pytest tests/test_audit_plan_doc.py
python -m py_compile scripts/audit_plan_doc.py tests/test_audit_plan_doc.py
python scripts/audit_plan_doc.py plans/PR-Audit-Plan-Doc-Shape.md
git diff --check
```

## Estimated diff size

| File | LOC (approx) |
|---|---:|
| `scripts/audit_plan_doc.py` | 105 |
| `tests/test_audit_plan_doc.py` | 185 |
| `plans/PR-Audit-Plan-Doc-Shape.md` | 72 |
| `docs/extraction/coordination/inflight.md` | 2 |
| **Total** | **~364** |

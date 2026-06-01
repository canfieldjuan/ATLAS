# PR-Invoicing-Approval-Blockers-Import-Cycles
## Why this slice exists
Worktree cleanup found a preserved clean branch,
`codex/invoicing-approval-blockers-preserved`, with one unmerged commit that
fixed invoice approval blockers and import cycles. Current `origin/main`
already absorbed part of that work, but the remaining gaps still matter:
blocked draft invoices import PDF/email machinery before they can be skipped,
some package `__init__` modules eagerly import settings-dependent modules, and
updating a monthly placeholder invoice with billable hours does not clear the
`metadata.needs_hours` blocker.
## Scope (this PR)
Ownership lane: invoicing
Slice phase: Production hardening
1. Keep the send path from importing PDF/email dependencies until after status,
   `needs_hours`, email, and dry-run blockers pass.
2. Lazily resolve remaining package-level exports that can pull
   settings-dependent modules during imports.
3. Clear `metadata.needs_hours` only when explicit line-item updates make every
   line billable, and recalculate replacement line-item amounts before the
   draft becomes sendable.
4. Add focused tests for the blocker-clearing helper and invoice update path.
### Files touched
- `plans/PR-Invoicing-Approval-Blockers-Import-Cycles.md`
- `atlas_brain/mcp/invoicing_server.py`
- `atlas_brain/services/__init__.py`
- `atlas_brain/storage/repositories/invoice.py`
- `atlas_brain/templates/email/__init__.py`
- `.github/workflows/atlas_invoicing_checks.yml`
- `tests/test_monthly_invoice_generation.py`
## Mechanism
The old preserved commit is ported selectively onto current main. The invoice
repository adds a small `_line_items_are_billable(...)` predicate and, when
`update_invoice(...)` receives replacement `line_items`, clears an existing
`metadata.needs_hours` flag only if every line has positive quantity and unit
price. Explicit replacement line items also have their `amount` fields
recalculated before persistence, so stale placeholder amounts cannot survive
into the email/PDF renderers after the blocker clears. The update SQL persists
the adjusted metadata with the recalculated totals.

`approve_and_send(...)` keeps the existing blocker order but moves PDF,
settings, email-provider, and invoice-email-template imports to the point where
a non-dry-run invoice is actually about to be sent. The services and email
template package exports use `__getattr__` for the settings-dependent optional
exports, mirroring the lazy `atlas_brain.auth` pattern that already exists on
current main.
## Intentional
- The current main already has lazy `atlas_brain.auth` exports and
  case-insensitive invoice number lookup, so this PR does not reapply those
  parts of the preserved branch.
- No migration is needed; invoice metadata is already JSONB.
- No email send behavior changes for invoices that pass blockers; imports move
  later, but the rendered PDF/email path is unchanged.
## Deferred
- Future PR: decide whether the preserved `codex/invoicing-approval-blockers-preserved`
  branch can be deleted after this PR lands.
Parked hardening: none.
## Verification
- Py compile passed for the changed Python files:
  `python -m py_compile ...`
- Focused pytest passed:
  `pytest tests/test_monthly_invoice_generation.py -k 'update_invoice_clears_needs_hours_when_line_items_are_billable or line_items_are_billable_requires_all_positive_quantities' -q`
  Result: 2 passed, 41 deselected.
- Local PR review passed with the staged PR body:
  `bash scripts/local_pr_review.sh --current-pr-body-file <body-file>`
## Estimated diff size
| Area | LOC |
|---|---:|
| Plan | 75 |
| CI workflow | 45 |
| Invoicing/runtime code | 120 |
| Tests | 85 |
| **Total** | **325** |

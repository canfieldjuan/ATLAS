# Invoicing Number Case-Insensitive Lookup

## Why this slice exists

The live draft-writer connector smoke created a draft invoice and returned the
invoice number `INV-2026-May-0185`, but the exposed `get_invoice` tool could
not find that invoice by number. The repository generates invoice numbers with
`to_char(..., 'YYYY-Mon')`, preserving mixed-case month names, while
`InvoiceRepository.get_by_number()` uppercases the caller input before an exact
SQL comparison. That turns `INV-2026-May-0185` into `INV-2026-MAY-0185` and
misses the row.

This breaks the connector contract: a host-facing tool can create a draft and
return an invoice number that the same tool surface cannot read back.

## Scope

1. Make invoice-number lookup tolerant of caller casing while preserving exact
   invoice-number storage.
2. Add a focused repository regression proving mixed-case generated invoice
   numbers remain retrievable.
3. Keep the draft-writer MCP surface unchanged.

### Files touched

- `atlas_brain/storage/repositories/invoice.py`
- `tests/test_invoice_repository.py`
- `plans/PR-Invoicing-Number-Case-Insensitive-Lookup.md`

## Mechanism

`InvoiceRepository.get_by_number()` will stop mutating the lookup argument to
uppercase. It will compare `lower(invoice_number) = lower($1)` in SQL and pass
the caller-provided invoice number after trimming surrounding whitespace.

The regression uses a fake asyncpg-shaped pool against the real repository
method, captures the SQL and bind argument, and returns a row only if the query
uses a case-insensitive comparison. That locks the behavior without requiring a
live database.

## Intentional

- No migration: invoice numbers already exist in mixed case and do not need to
  be rewritten.
- No change to invoice generation: existing `YYYY-Mon` numbers stay stable for
  customer-facing invoices and previously sent PDFs/emails.
- No MCP tool change: the bug is below both read-only and draft-writer MCP
  servers, so fixing the repository closes both surfaces.
- No cleanup of the live smoke draft: it is blocked from sending by missing
  email and zero subtotal and remains useful as an audit artifact.

## Deferred

- A reusable live write smoke script for the draft-writer connector can be a
  follow-up if we want repeatable operator verification after future MCP server
  changes.
- Any invoice-number format simplification is out of scope because it would
  affect historical customer-facing identifiers.

## Verification

- Focused pytest: `.venv/bin/python -m pytest` against the repository
  regression and draft-writer MCP suite; result: 14 passed.
- Python compile check for the repository and regression test; result: passed.
- Git whitespace check; result: passed.
- Local PR review bundle in advisory dirty mode; result: passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Repository lookup patch | ~5 |
| Regression test | ~60 |
| Plan doc | ~70 |
| **Total** | ~135 |

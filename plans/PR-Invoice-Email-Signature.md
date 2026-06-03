# PR-Invoice-Email-Signature

## Why this slice exists

Invoice cover emails -- both the message the monthly auto-invoice task sends and
the body the operator reviews as a Gmail draft -- signed off with only the bare
business name, phone, and email. The operator wants a fuller, consistent
sign-off that includes the sender name and title and uses the operator's contact
address.

This slice replaces the cover-email sign-off with a single shared signature
constant so every invoice email, current and future, carries the same block. It
does not change invoice amounts, the formal invoice document, or any send
behavior.

## Scope (this PR)

Ownership lane: invoicing/cover-email-signature
Slice phase: Product polish

1. Add a BUSINESS_SIGNATURE constant to the invoice email template module.
2. Use it as the sign-off in the monthly auto-invoice cover email, replacing the
   bare name/phone/email lines.

### Files touched

- `atlas_brain/templates/email/invoice.py`
- `atlas_brain/autonomous/tasks/monthly_invoice_generation.py`
- `plans/PR-Invoice-Email-Signature.md`

## Mechanism

`atlas_brain/templates/email/invoice.py` gains a BUSINESS_SIGNATURE string
constant holding the operator sign-off block (name and title, business name,
address, phone, contact email).
`atlas_brain/autonomous/tasks/monthly_invoice_generation.py` imports
BUSINESS_SIGNATURE and interpolates it into the plain-text cover-email body in
place of the previous BUSINESS_NAME / BUSINESS_PHONE / BUSINESS_EMAIL sign-off
lines. The "Make all checks payable to" line still uses BUSINESS_NAME.

## Intentional

- Email cover-note sign-off only. The formal invoice document footer rendered by
  render_invoice_html / render_invoice_text is unchanged and still shows the
  business contact block.
- No change to invoice amounts, due dates, line items, or send/review behavior.
- The signature uses the operator Gmail as the contact address by request; the
  business email constant stays in place for the formal invoice document.

## Deferred

- Future PR: apply the same signature to the formal invoice document footer if
  the operator wants the PDF/document to match.
- Parked hardening: none.

## Verification

- pytest for `tests/test_monthly_invoice_generation.py` -- 43 passed.
- python import of `atlas_brain/autonomous/tasks/monthly_invoice_generation.py`
  and `atlas_brain/templates/email/invoice.py` -- resolves; signature renders as
  specified.
- check_ascii_python.sh -- added lines are ASCII-only.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~75 |
| Template constant | ~8 |
| Task body | ~6 |
| **Total** | **~89** |

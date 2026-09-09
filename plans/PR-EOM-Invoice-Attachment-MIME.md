# PR-EOM-Invoice-Attachment-MIME

## Why this slice exists

A commercial customer replied to an August invoice with "Can you please resend
this invoices? It is not showing up." The message had been delivered, the
attachment was present, and the PDF on disk was structurally sound: its xref
offsets matched the real object positions byte for byte and `startxref` pointed
at the xref table. The sender could open the same file locally.

Inspecting the raw MIME of the delivered message showed the attachment part
declared `Content-Type: application/octet-stream` while its
`Content-Disposition` filename ended in `.pdf`. Both Gmail send paths hardcode
`MIMEBase("application", "octet-stream")` for every attachment. Recipient mail
gateways routinely strip or quarantine generic binary streams, which makes an
attachment disappear for the recipient while the message itself arrives intact
and looks correct in the sender's own mailbox and in the Gmail API's own
`attachments[].mimeType` view.

This is a delivery-correctness defect in the transport, not a document defect.

### Problem-derived contract

- Root cause: the Gmail transport assumes a single attachment type rather than
  deriving it, so the declared `Content-Type` cannot ever agree with the
  filename for anything that is not an octet stream.
- Correct fix: resolve the type at the one place the part is constructed --
  explicit caller value first, filename extension second, octet-stream only as
  a last resort -- and apply it to both send paths, since either can carry an
  invoice.
- Must not change: OAuth handling, the send/draft request shapes, header
  validation and the protected-header allowlist, retry and ambiguous-status
  handling, or the body text of any message that does not opt into a note.

## Scope (this PR)

Ownership lane: eom/invoice-attachment-mime
Slice phase: production hardening
Max files: 4

1. Derive the attachment MIME type in both Gmail send paths instead of
   hardcoding octet-stream.
2. Declare `application/pdf` explicitly on the invoice attachment, so the
   correct type does not depend on extension inference alone.
3. Widen `approve_and_send`'s `invoice_ids` to accept a list, and add an
   optional `note` for one-off context above the standard body.
4. Prove both directions of the type resolution with a focused test.

### Review Contract

- A `.pdf` attachment is never built as `application/octet-stream`; settled by
  `test_a_pdf_attachment_is_never_built_as_octet_stream` and
  `test_pdf_extension_is_inferred_when_no_type_is_declared`.
- An explicit `mime_type` overrides a misleading extension; settled by
  `test_explicit_mime_type_wins_over_the_extension`.
- Input that cannot resolve still yields a valid pair rather than an empty or
  malformed subtype; settled by
  `test_unresolvable_filename_falls_back_to_octet_stream` and
  `test_declared_type_without_a_subtype_does_not_yield_an_empty_subtype`.
- An empty or absent declared type defers to the extension rather than being
  treated as a declaration; settled by
  `test_empty_declared_type_defers_to_the_extension`.
- `approve_and_send` accepts both the JSON-array-string and the list form of
  `invoice_ids`; the body already branched on `isinstance(invoice_ids, str)`,
  so only the annotation changes.
- A send with no `note` produces a body byte-identical to today's.
- Affected surfaces: every Gmail-delivered attachment in the repo, not only
  invoices. Risk areas: a caller that depended on receiving octet-stream, and
  filename extensions that resolve to an unexpected type.
- Reviewer rules triggered: R1, R2, R5, R10.

### Boundary-change enumeration and closure declaration

- Decision seam: `_attachment_type` is the only place an attachment's declared
  type is chosen; both send paths call it.
- Inputs are OPEN: caller-supplied `mime_type` and caller-supplied filenames.
  The resolution order is CLOSED and finite -- explicit value, then
  `mimetypes.guess_type`, then the octet-stream fallback.
- The previous behavior is preserved exactly for anything that resolves to
  nothing, so an unknown attachment is unchanged. Everything that resolves is
  newly DECLARED rather than defaulted, which is the safe side: a wrong-but-
  specific type is visible to the recipient, whereas the previous blanket
  default was silently discarded by gateways.
- A declared value with no `/` yields `octet-stream` rather than an empty
  subtype, because an empty subtype produces a malformed header.

### Deployed-config probing

No configuration, environment, or credential changes. `gmail_send_enabled`
selects this transport over Resend and is untouched. `mimetypes` uses the
stdlib table with no registry additions, so resolution does not depend on host
MIME configuration for the extensions this repo sends.

### Files touched

- `plans/PR-EOM-Invoice-Attachment-MIME.md`
- `atlas_brain/tools/gmail.py`
- `atlas_brain/mcp/invoicing_server.py`
- `tests/test_gmail_attachment_mime.py`

## Mechanism

`_attachment_type` takes the attachment mapping and its filename and returns a
`(maintype, subtype)` pair for `MIMEBase`. Both send paths call it in place of
the hardcoded constructor; nothing else about part construction, the
`Content-Disposition` header, or the base64 payload changes.

`approve_and_send` passes `mime_type` explicitly so the invoice path does not
rely on extension inference, and its `invoice_ids` annotation is widened to
match the list handling its body already performed.

## Intentional

- No per-caller allowlist of acceptable types: the resolution rule, not an
  enumeration of known filenames, is the durable source.
- No change to the octet-stream fallback itself, which remains correct for
  genuinely unknown bytes.
- `note` is omitted entirely rather than defaulted to an empty line, so the
  standard body is unchanged for every existing caller.

## Deferred

Parking predicate: transport behavior not implicated by the observed delivery
failure is parked rather than changed speculatively.

Parked hardening: the Resend path's attachment typing was not inspected as
part of this diff; `send_invoice`'s inline-HTML path renders a tax row with a
hardcoded `Tax:` label while `invoice_pdf.py` honors `metadata.tax_label`.
Both are recorded here rather than folded in.

## Verification

Focused test file, byte-compilation of both changed modules, and the existing
Gmail and invoicing suites (`test_commercial_billing_gmail_drafts.py`,
`test_eom_scoped_gmail_credentials.py`, `test_eom_scoped_gmail_hardening.py`,
`test_invoicing_draft_writer_mcp.py`, `test_invoicing_readonly_mcp.py`,
`test_monthly_invoice_generation.py`). Do not run the broad Unit Gate locally;
GitHub owns its full suite. Rollback is a revert of both commits; the change
affects message headers only and touches no persistent data.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-EOM-Invoice-Attachment-MIME.md` | 145 |
| `atlas_brain/tools/gmail.py` | 18 |
| `atlas_brain/mcp/invoicing_server.py` | 10 |
| `tests/test_gmail_attachment_mime.py` | 63 |
| **Total** | **236** |

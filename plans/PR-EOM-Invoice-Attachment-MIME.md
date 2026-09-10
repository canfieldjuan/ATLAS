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
  invoice. Because the explicit value is caller input that lands in a raw MIME
  header, it is admitted only when it is one legal `type/subtype` pair; anything
  else refuses the send before any request, the same way this transport already
  treats caller-supplied headers.
- Must not change: OAuth handling, the send/draft request shapes, header
  validation and the protected-header allowlist, retry and ambiguous-status
  handling, or the body text of any message that does not opt into a note.

## Scope (this PR)

Ownership lane: eom/invoice-attachment-mime
Slice phase: production hardening
Max files: 5

1. Derive the attachment MIME type in both Gmail send paths instead of
   hardcoding octet-stream, admitting a declared type only when it is one legal
   pair.
2. Declare `application/pdf` explicitly on the invoice attachment, so the
   correct type does not depend on extension inference alone.
3. Widen `approve_and_send`'s `invoice_ids` to accept a list, keep an explicit
   selection -- including an empty one -- from widening into "every draft", and
   add an optional `note` for one-off context above the standard body.
4. Prove the type resolution, the selection rule, and the body through the real
   entrypoints: the transport's posted raw message and the MCP tool against the
   real invoice repository.

### Review Contract

- A `.pdf` attachment is never delivered as `application/octet-stream`; settled
  by `test_send_infers_the_pdf_from_the_filename_when_nothing_is_declared`
  (decodes the raw message posted to Gmail) and
  `test_pdf_extension_is_inferred_when_no_type_is_declared`.
- An explicit `mime_type` overrides a misleading extension and reaches the raw
  message on both paths; settled by
  `test_explicit_mime_type_wins_over_the_extension`,
  `test_send_declares_the_pdf_in_the_raw_message_gmail_receives` and
  `test_create_draft_declares_the_pdf_in_the_raw_message_gmail_receives`.
- Every legal pair is admitted and case-folded, and every declaration outside
  the pair grammar (CR/LF injection, parameters, a second slash, an empty side,
  whitespace, NUL, non-ASCII, over-length, non-string) is refused with the
  transport's definitely-not-sent / definitely-not-created error before any
  HTTP request; settled by `test_every_legal_pair_is_admitted_and_case_folded`,
  `test_every_malformed_declaration_is_refused_not_repaired`,
  `test_a_non_string_declaration_is_refused`,
  `test_send_refuses_a_malformed_declaration_before_any_request` and
  `test_create_draft_refuses_a_malformed_declaration_before_any_request`.
- An empty or absent declared type defers to the extension, and an
  unresolvable filename lands on octet-stream; settled by
  `test_empty_declared_type_defers_to_the_extension` and
  `test_unresolvable_filename_falls_back_to_octet_stream`.
- `approve_and_send` accepts both the JSON-array-string and the list form of
  `invoice_ids`, sends exactly the named invoice, and declares the PDF; settled
  by `test_both_selection_forms_send_exactly_the_named_invoice_as_a_declared_pdf`.
- An explicit empty selection (`[]` or `"[]"`) sends nothing and touches no
  invoice; only an omitted argument covers every matching draft; a selection
  that is not a list of strings is refused without sending; settled by
  `test_an_explicit_empty_selection_sends_nothing`,
  `test_an_omitted_selection_still_covers_every_matching_draft` and
  `test_a_selection_that_is_not_a_list_of_strings_is_refused_and_sends_nothing`.
- A send with no `note`, or a blank one, produces a body byte-identical to
  today's, checked against a body written out independently of the tool; a
  note is stripped and placed above it; settled by
  `test_without_a_note_the_body_is_byte_identical_to_the_standard_body`,
  `test_a_blank_note_leaves_the_body_unchanged` and
  `test_a_note_is_placed_above_the_standard_body_and_stripped`.
- Affected surfaces: every Gmail-delivered attachment in the repo, not only
  invoices. Risk areas: a caller that depended on receiving octet-stream, a
  caller passing a malformed `mime_type` (none exists in the repo; the only
  caller passes the constant `application/pdf`), and filename extensions that
  resolve to an unexpected type.
- Reviewer rules triggered: R1, R2, R5, R10.

### Boundary-change enumeration and closure declaration

- Decision seam: `_attachment_type` is the only place an attachment's declared
  type is chosen; both send paths call it.
- Inputs are OPEN: caller-supplied `mime_type` and caller-supplied filenames.
  The safety decision is allowlist-shaped and CLOSED: a non-empty declaration
  is admitted only on `fullmatch` of the RFC 6838 restricted-name grammar
  (`name/name`, each one alphanumeric then up to 126 of `A-Za-z0-9!#$&^_.+-`),
  and everything else -- not a subset of known-bad strings -- refuses the whole
  send or draft before any request. The declaration is never repaired, because
  a repaired header silently changes what the caller asked for.
- An empty or absent declaration is not a declaration: the filename extension
  is inferred, the inferred value passes the same recognizer, and only when
  nothing resolves does the pair fall back to `application/octet-stream`. So an
  unknown attachment is unchanged, and everything that resolves is newly
  DECLARED rather than defaulted.
- Selection seam in `approve_and_send`: `None` means omitted and covers every
  draft matching `status_filter`; any other value is a selection, must be a
  list of strings (a JSON string is parsed first), and an empty list selects
  nothing. The list form and the string form reach the same rule.

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
- `tests/test_invoicing_approve_and_send_selection.py`

## Mechanism

`_attachment_type` takes the attachment mapping and its filename and returns a
`(maintype, subtype)` pair for `MIMEBase`, raising `ValueError` for a
declaration outside the pair grammar. Both send paths call it in place of the
hardcoded constructor and convert that `ValueError` into `GmailSendError`
(`definitely_not_sent=True`) or `GmailDraftCreateError`
(`definitely_not_created=True`), mirroring how invalid extra headers are
already handled; nothing else about part construction, the
`Content-Disposition` header, or the base64 payload changes.

`approve_and_send` passes `mime_type` explicitly so the invoice path does not
rely on extension inference, widens its `invoice_ids` annotation to the list
its body already handled, distinguishes an omitted argument from an explicit
selection, and prepends a stripped `note` only when one is supplied.

## Intentional

- No per-caller allowlist of acceptable types: the pair grammar, not an
  enumeration of known filenames or types, is the durable source.
- A malformed declaration refuses rather than falls back. Falling back would
  hide a caller bug and turn a header-injection attempt into a silent
  octet-stream, which is the exact blanket default this slice removes.
- No change to the octet-stream fallback itself, which remains correct for
  genuinely unknown bytes.
- `note` is omitted entirely rather than defaulted to an empty line, so the
  standard body is unchanged for every existing caller.
- The selection tests substitute only the outbound email port with a capturing
  provider; the repository, PDF renderer, and status transitions are real. The
  transport tests prove the captured `mime_type` reaches Gmail's raw message.

## Deferred

Parking predicate: transport behavior not implicated by the observed delivery
failure is parked rather than changed speculatively.

Parked hardening: the Resend path's attachment typing was not inspected as
part of this diff; `send_invoice`'s inline-HTML path renders a tax row with a
hardcoded `Tax:` label while `invoice_pdf.py` honors `metadata.tax_label`.
Both are recorded here rather than folded in.

## Verification

Focused test files (transport-level and MCP-level), byte-compilation of both
changed modules, the ASCII gate, the `atlas_brain/tools` and `atlas_brain/mcp`
maturity-sweep ratchet lanes as CI runs them, the pre-push audit, and the
existing Gmail and invoicing suites (`test_commercial_billing_gmail_drafts.py`,
`test_eom_scoped_gmail_credentials.py`, `test_eom_scoped_gmail_hardening.py`,
`test_invoicing_draft_writer_mcp.py`, `test_invoicing_readonly_mcp.py`,
`test_monthly_invoice_generation.py`). The selection tests need the local
Postgres the existing invoicing tests already use; they create their own draft
invoices under a unique `source_ref` and void them afterwards. Do not run the
broad Unit Gate locally; GitHub owns its full suite. Rollback is a revert of
the commits; the change affects message headers and the tool's selection rule
only and touches no persistent data beyond the tests' own voided rows.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-EOM-Invoice-Attachment-MIME.md` | 200 |
| `atlas_brain/tools/gmail.py` | 50 |
| `atlas_brain/mcp/invoicing_server.py` | 36 |
| `tests/test_gmail_attachment_mime.py` | 234 |
| `tests/test_invoicing_approve_and_send_selection.py` | 165 |
| **Total** | **685** |

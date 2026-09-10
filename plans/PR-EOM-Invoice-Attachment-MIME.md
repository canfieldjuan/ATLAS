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

This slice is over the 400-LOC target and is not split, on purpose. The
runtime change is small (the transport, the fallback classification in the
composite provider and the email tool, and the MCP tool's selection rule);
the overage is the proof. A guard on a caller-declared value that lands in a
raw MIME header must ship with its grammar-derived class-closure test, its
transport-level proof through the posted raw message, its production-
entrypoint proof that a refusal never falls back to Resend, and, for the
money-path selection rule, its MCP-boundary proof on the real repository.
Landing the guard first and the proofs later would merge an unverified guard
on customer mail; landing the proofs first would test code that does not
exist. The plan doc itself is a further part of the overage.

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
  treats caller-supplied headers. The composite provider and the email tool
  caught every exception from the Gmail attempt as an outage and retried
  through Resend with the same input; that fallback is the defect that would
  have swallowed the refusal, so the refusal is classified once at its source
  (a `GmailInvalidInput` error) and the fallback paths do not fall back on it.
- Must not change: OAuth handling, the send/draft request shapes, header
  validation and the protected-header allowlist, retry and ambiguous-status
  handling, or the body text of any message that does not opt into a note.

## Scope (this PR)

Ownership lane: eom/invoice-attachment-mime
Slice phase: production hardening
Max files: 12

1. Derive the attachment MIME type in both Gmail send paths instead of
   hardcoding octet-stream, admitting a declared type only when it is one legal
   pair, treating an explicit `application/octet-stream` as no declaration, and
   raising a distinct input error that the composite provider and the email
   tool never fall back from, so a refused input can never be retried through
   Resend.
2. Declare `application/pdf` explicitly on the invoice attachment, so the
   correct type does not depend on extension inference alone.
3. Widen `approve_and_send`'s `invoice_ids` to accept a list, keep an explicit
   selection -- including an empty one -- from widening into "every draft",
   reject an explicit JSON `null` at the MCP boundary (the annotation is
   deliberately not `Optional`, so only a genuinely omitted argument means
   "every matching draft"), and add an optional `note` for one-off context
   above the standard body.
4. Prove the type resolution, the selection rule, and the body through the real
   entrypoints: the transport's posted raw message, and the published MCP tool
   (`mcp.call_tool` / `mcp.list_tools`) against the real invoice repository on
   an isolated schema, enrolled in the Postgres-backed invoicing workflow.
5. Stop the unit gate's MCP stub leak at its source. Eight b2b/content-ops test
   modules install a `MagicMock` as `mcp.server.fastmcp` via
   `sys.modules.setdefault` at import time, so in the gate's single full-suite
   process every MCP test module collected after them fails to import; the
   repository had baselined six such files as permanent collection errors, and
   the new selection test became the seventh. `tests/conftest.py` now
   pre-imports the real SDK before collection -- the same guard it already
   applies to `asyncpg` -- which makes those `setdefault` calls no-ops, and
   `tests/unit_gate_baseline.txt` shrinks by those six file entries. The two
   tests in `test_mcp_content_ops_marketer_verify.py` that genuinely failed
   once the file ran are fixed in place, because the gate accepts no baseline
   additions.
   A second leak surfaced once those files ran in the full process:
   `tests/test_auth_api_keys.py` reloads `atlas_brain.config` and leaves a new
   `settings` object bound, so later tests that patch the import-time
   `settings` no longer reach code that imports it lazily (the content-ops
   HTTP-auth tests). An autouse fixture there restores the config module's
   globals after each of its tests.

### Review Contract

- A `.pdf` attachment is never delivered as `application/octet-stream`, not
  even when a caller declares that generic type explicitly; settled by
  `test_send_infers_the_pdf_from_the_filename_when_nothing_is_declared`
  (decodes the raw message posted to Gmail),
  `test_pdf_extension_is_inferred_when_no_type_is_declared` and
  `test_an_explicit_octet_stream_declaration_does_not_defeat_pdf_inference`.
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
- The refusal holds through the production entrypoints: the real
  `CompositeEmailProvider.send` re-raises the Gmail input error instead of
  falling back, the real `EmailTool` returns `INVALID_PARAMETER`, in both
  cases with no request to the Gmail edge and no fallback to Resend, a forced
  Resend send still goes only to Resend, and a well-declared PDF still reaches
  Gmail's raw message through both; settled by
  `test_composite_provider_refuses_a_malformed_declaration_before_choosing_a_provider`,
  `test_composite_provider_forced_resend_still_goes_to_resend_only`,
  `test_composite_provider_delivers_a_declared_pdf_through_the_real_gmail_transport`,
  `test_email_tool_refuses_a_malformed_declaration_without_trying_either_transport`
  and `test_email_tool_delivers_a_declared_pdf_through_the_real_gmail_transport`.
- An empty or absent declared type defers to the extension, and an
  unresolvable filename lands on octet-stream; settled by
  `test_empty_declared_type_defers_to_the_extension` and
  `test_unresolvable_filename_falls_back_to_octet_stream`.
- The published `approve_and_send` schema accepts `invoice_ids` as a string or
  an array of strings, both optional, and an optional string `note`; settled by
  `test_the_published_tool_schema_accepts_a_list_or_a_json_string_and_an_optional_note`
  through `mcp.list_tools`.
- Through `mcp.call_tool`, both the JSON-array-string and the list form of
  `invoice_ids` send exactly the named invoice and declare the PDF; settled by
  `test_both_selection_forms_send_exactly_the_named_invoice_as_a_declared_pdf`.
- An explicit empty selection (`[]` or `"[]"`) sends nothing and touches no
  invoice; only an omitted argument covers every matching draft; a string that
  is not a JSON array of strings is refused by the tool without sending; a
  value outside the schema (an explicit `null`, ints in the list, `None` in
  the list, a bare number, an object, a JSON-object string) is refused at the
  boundary before the tool runs, and the published schema carries no null
  branch; settled by `test_an_explicit_empty_selection_sends_nothing`,
  `test_an_omitted_selection_still_covers_every_matching_draft`,
  `test_a_string_that_is_not_a_json_array_of_strings_is_refused_and_sends_nothing`
  and `test_the_boundary_rejects_a_selection_outside_the_schema_before_the_tool_runs`.
- A send with no `note`, or a blank one, produces a body byte-identical to
  today's, checked against a body written out independently of the tool; a
  note is stripped and placed above it; settled by
  `test_without_a_note_the_body_is_byte_identical_to_the_standard_body`,
  `test_a_blank_note_leaves_the_body_unchanged` and
  `test_a_note_is_placed_above_the_standard_body_and_stripped`; a non-string
  note is refused at the boundary, settled by
  `test_a_note_outside_the_schema_is_rejected_at_the_boundary`.
- The unit gate collects every MCP test module in one process: the full-suite
  collection under the gate's own shape (`-m "not integration and not e2e"`,
  no database) reports no `mcp` stub errors, and `scripts/check_unit_gate.py`
  itself passes against the shrunken baseline; settled by running that script
  locally under the gate's shape and pins, recorded in the PR body's
  mechanical verification.
- The Resend route never receives the port's Gmail-side `mime_type`: the real
  `EmailTool` translates each attachment into Resend's `{filename, content,
  content_type}` shape, through the tool directly (Gmail unavailable) and
  through the real composite-to-Resend route (`provider="resend"`), and omits
  `content_type` when nothing was declared; settled by
  `test_email_tool_translates_mime_type_into_resend_content_type`,
  `test_composite_forced_resend_reaches_resend_with_content_type_not_mime_type`
  and `test_resend_route_omits_content_type_when_nothing_was_declared`.
- Affected surfaces: every Gmail-delivered attachment in the repo, not only
  invoices. Risk areas: a caller that depended on receiving octet-stream, a
  caller passing a malformed `mime_type` (none exists in the repo; the only
  caller passes the constant `application/pdf`), and filename extensions that
  resolve to an unexpected type.
- A structured major type (`message/*`, `multipart/*`), declared or inferred
  (`.eml`, `.mht`), keeps `application/octet-stream`, because this builder emits
  leaf `MIMEBase` parts and a container type would make the recipient's parser
  nest a message instead of returning the bytes; settled by
  `test_a_structured_major_type_keeps_octet_stream_because_a_leaf_part_cannot_carry_it`
  and `test_send_delivers_an_eml_attachment_whose_bytes_come_back_intact`
  (decodes the posted raw message and compares the payload bytes).
- Reviewer rules triggered: R1, R2, R3, R5, R8, R10, R13, R14.
  - R3 (security, input trust): the declared `mime_type` is caller input that
    lands in a raw MIME header; it is admitted only on recognition of the
    pair grammar and refused otherwise, with CR/LF/NUL injection in the
    refused set, before any request and before any base64 decoding; the
    refusal is classified so no fallback retries it through Resend. The MCP
    `invoice_ids` argument is validated at the FastMCP boundary (no null
    branch, list of strings only). No secret, token, or credential path
    changes.
  - R8 (idempotency, double-send): `approve_and_send` still sends one message
    per draft and marks it `sent` in the same iteration, so a retry of the
    tool call finds `status != draft` and skips; an empty or refused
    selection performs no write; the Gmail-then-Resend fallback cannot send
    the same message through both providers on a refused input because the
    refusal is not eligible for fallback, and a Gmail outage still falls
    back exactly once as before. Settled by
    `test_both_selection_forms_send_exactly_the_named_invoice_as_a_declared_pdf`
    (the untouched draft stays `draft`) and the composite tests.
  - R13 (class, not example): every Codex example is covered by a generated
    class, not a fixture: the grammar-derived closure tests generate legal
    names and fifteen mutations across families and containers; the
    encoded-suffix and structured-major rules are tested on suffixes and
    types the review did not name (`.svgz`, `.bz2`, `.Z`, `.mht`,
    `Message/Partial`, `multipart/mixed`).
  - R14 (boundary probe, both directions): admitted side, generated legal
    pairs and every inferred family; refused side, generated mutations
    including empty type, empty subtype, over-length, non-string, and the
    combined bad-declaration-plus-bad-base64 case; representation parity
    across single, mixed-before, and mixed-after containers. The selection
    guard is probed with `[]`, `"[]"`, `null`, ints, `None` in a list, a
    bare number, an object, and a JSON-object string, plus the omitted case
    that must still cover every draft.

### Boundary-change enumeration and closure declaration

- Decision seam: `_attachment_type` is the only place an attachment's declared
  type is chosen; `validate_attachment_types` runs it over the whole list once
  at the top of `send` and `create_draft`. A refusal is raised as
  `GmailSendInputError` / `GmailDraftInputError` (both `GmailInvalidInput`),
  the same class as invalid caller headers, and the two fallback paths
  (`CompositeEmailProvider.send`, `EmailTool._try_gmail_send`) re-raise or
  return a failed result on that class instead of retrying through Resend.
  There is one decision point and one classification; no layer re-validates.
- Inputs are OPEN: caller-supplied `mime_type` and caller-supplied filenames.
  The safety decision is allowlist-shaped and CLOSED: a non-empty declaration
  is admitted only on `fullmatch` of the RFC 6838 restricted-name grammar
  (`name/name`, each one alphanumeric then up to 126 of `A-Za-z0-9!#$&^_.+-`),
  and everything else -- not a subset of known-bad strings -- refuses the whole
  send or draft before any request. The declaration is never repaired, because
  a repaired header silently changes what the caller asked for.
- Class closure (docs/GUARD_CLASS_CLOSURE.md req 3): legal names are generated
  from the grammar and crossed with fifteen mutations that leave it, three
  declaration families, and three container shapes handed to
  `validate_attachment_types`; every verdict is checked against
  `_expected_verdict`, an oracle written from this contract rather than from
  the guard. Settled by
  `test_every_generated_legal_pair_is_admitted_as_the_oracle_says`,
  `test_every_generated_mutation_is_refused_and_containers_do_not_change_the_verdict`
  and
  `test_no_declaration_and_octet_stream_declaration_defer_to_the_filename_across_containers`.
- An empty, absent, or `application/octet-stream` declaration is not a
  declaration (the generic type carries no information and is the blanket
  default this slice replaces): the filename extension is inferred, the inferred value passes the same recognizer, and only when
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

- `.github/workflows/atlas_invoicing_checks.yml`
- `atlas_brain/mcp/invoicing_server.py`
- `atlas_brain/services/email_provider.py`
- `atlas_brain/tools/email.py`
- `atlas_brain/tools/gmail.py`
- `plans/PR-EOM-Invoice-Attachment-MIME.md`
- `tests/conftest.py`
- `tests/test_auth_api_keys.py`
- `tests/test_gmail_attachment_mime.py`
- `tests/test_invoicing_approve_and_send_selection.py`
- `tests/test_mcp_content_ops_marketer_verify.py`
- `tests/unit_gate_baseline.txt`

## Mechanism

`_attachment_type` takes the attachment mapping and its filename and returns a
`(maintype, subtype)` pair for `MIMEBase`, raising `ValueError` for a
declaration outside the pair grammar and treating an explicit
`application/octet-stream` as no declaration. `send` and `create_draft` run
`validate_attachment_types` over the whole list up front, next to the
existing header validation, and raise `GmailSendInputError` /
`GmailDraftInputError` -- subclasses of the existing errors carrying
`definitely_not_sent` / `definitely_not_created`, and of the `GmailInvalidInput`
marker. `CompositeEmailProvider.send` re-raises `GmailInvalidInput` before its
generic outage fallback; `EmailTool._try_gmail_send` returns an
`INVALID_PARAMETER` result on it instead of `None` (which means "fall back").
On the Resend path, `EmailTool._resend_attachment` maps the port's
`mime_type` to Resend's `content_type` and drops nothing else, so a Gmail
fallback never forwards a property Resend does not define. Both send paths call it in place of the
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
  provider; the repository, PDF renderer, and status transitions are real, on
  a throwaway schema built from the invoice migrations inside the receivables
  test database. Every case crosses the published MCP boundary, because the
  defect lived in the annotation FastMCP validates against, not in the body.
  The transport tests prove the captured `mime_type` reaches Gmail's raw
  message.

## Deferred

Parking predicate: transport behavior not implicated by the observed delivery
failure is parked rather than changed speculatively.

Parked hardening: the Resend path's attachment typing was not inspected as
part of this diff; `send_invoice`'s inline-HTML path renders a tax row with a
hardcoded `Tax:` label while `invoice_pdf.py` honors `metadata.tax_label`; and
`GmailEmailProvider.send` does not forward caller `headers`, so a composite
caller passing them gets a `TypeError` that the generic fallback still retries
through Resend (headers are a direct-transport feature today). All three are
recorded here rather than folded in.

## Verification

Focused test files (transport-level and MCP-level), byte-compilation of both
changed modules, the ASCII gate, the `atlas_brain/tools` and `atlas_brain/mcp`
maturity-sweep ratchet lanes as CI runs them, the pre-push audit, and the
existing Gmail and invoicing suites (`test_commercial_billing_gmail_drafts.py`,
`test_eom_scoped_gmail_credentials.py`, `test_eom_scoped_gmail_hardening.py`,
`test_invoicing_draft_writer_mcp.py`, `test_invoicing_readonly_mcp.py`,
`test_monthly_invoice_generation.py`). The selection tests are marked
`integration`, skip unless `ATLAS_RECEIVABLES_TEST_DATABASE_URL` is set, and
run in the Postgres-backed `atlas-invoicing-checks` job on a schema they create
and drop; the Unit Gate (`-m "not integration and not e2e"`, no database)
never collects them. Do not run the broad Unit Gate locally; GitHub owns its
full suite. Rollback is a revert of the commits; the change affects message
headers and the tool's selection rule only and touches no persistent data
beyond the tests' own dropped schema.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_invoicing_checks.yml` | 12 |
| `atlas_brain/mcp/invoicing_server.py` | 42 |
| `atlas_brain/services/email_provider.py` | 6 |
| `atlas_brain/tools/email.py` | 33 |
| `atlas_brain/tools/gmail.py` | 110 |
| `plans/PR-EOM-Invoice-Attachment-MIME.md` | 368 |
| `tests/conftest.py` | 13 |
| `tests/test_auth_api_keys.py` | 20 |
| `tests/test_gmail_attachment_mime.py` | 745 |
| `tests/test_invoicing_approve_and_send_selection.py` | 272 |
| `tests/test_mcp_content_ops_marketer_verify.py` | 30 |
| `tests/unit_gate_baseline.txt` | 6 |
| **Total** | **1657** |

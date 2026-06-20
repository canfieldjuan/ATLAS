# Support Ticket Provider Export Fixtures

Sanitized provider-shaped CSV fixtures for the FAQ deflection upload path.

These files are not live customer exports. They pin the column contracts the
upload UI advertises today so parser and report tests exercise realistic
provider shapes instead of one idealized internal schema.

## Contract

The deflection report treats provider exports according to the evidence they
carry:

- Full-thread exports can support publishable FAQ answers when they include
  customer-visible wording and public agent replies/resolutions.
- Ticket-list or ticket-index exports can still support clustering, gap-list
  preview, status/count diagnostics, and repeat-question analysis, but they do
  not prove answer text by themselves.
- Private or internal notes are intentionally ignored for customer wording and
  answer proof. They must not appear in customer-facing examples, unpaid
  previews, or generated FAQ copy.

The public `atlas-portfolio` intake validates and uploads raw source bytes, then
hands them to ATLAS. ATLAS owns parsing, admission diagnostics, private-field
exclusion, and the gap-list versus publishable-answer distinction.

Fixtures:

- `zendesk_full_thread_export.csv` -- Zendesk-style ticket export with requester
  comments and latest agent replies.
- `freshdesk_full_thread_export.csv` -- Freshdesk-style ticket export with
  ticket descriptions and agent replies.
- `help_scout_full_thread_export.csv` -- Help Scout-style conversation export
  with customer messages and support replies.
- `intercom_conversation_export.csv` -- Intercom-style conversation export
  with initial messages and admin replies.
- `zendesk_ticket_index_only.csv` -- ticket-list/index export without message
  bodies or replies; expected to remain gap-list-only in preview diagnostics.

## CI Proof

`tests/test_smoke_content_ops_support_ticket_package.py` pins this contract:

- `test_support_ticket_package_smoke_accepts_provider_full_thread_exports`
  verifies the full-thread fixtures ingest with customer wording and resolution
  evidence.
- `test_provider_full_thread_exports_generate_publishable_deflection_items`
  verifies those full-thread fixtures can generate resolution-evidence FAQ
  items.
- `test_support_ticket_package_smoke_marks_ticket_index_only_export_gap_list_only`
  verifies a ticket-index-only export stays gap-list-only.
- `test_support_ticket_package_uses_zendesk_public_comments_not_internal_notes`
  and `test_support_ticket_package_skips_private_comment_objects_in_history`
  verify private/internal note exclusion.

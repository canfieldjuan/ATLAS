# Support Ticket Provider Export Fixtures

Sanitized provider-shaped CSV fixtures for the FAQ deflection upload path.

These files are not live customer exports. They pin the column contracts the
upload UI advertises today so parser and report tests exercise realistic
provider shapes instead of one idealized internal schema.

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

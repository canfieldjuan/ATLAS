# Extracted Content Pipeline (Staging Copy)

This directory is an additive extraction scaffold copied from `atlas_brain`.
It is intentionally kept side-by-side with Atlas so pipeline logic can be
carved out safely without removing or changing production code.

## Current contents

- `autonomous/tasks/`: copied task implementations
- `services/`: copied support shims and staged service dependencies
- `skills/digest/`: copied prompt skill contracts, including campaign and
  sequence prompts used by the standalone services
- `storage/migrations/`: copied persistence migrations
- `docs/`: extraction maps for productized pipeline slices

## Sync command

To refresh this scaffold from Atlas source of truth:

```bash
bash scripts/sync_extracted_content_pipeline.sh
```

This stable product entry point delegates to
`extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline`.

## Manifest

Mirror mappings are declared in `extracted_content_pipeline/manifest.json` so sync and validation use one source of truth.

## Scope

This scaffold preserves code exactly as copied so behavior and signatures remain
unchanged while extraction work continues.

The standalone audit now passes with zero runtime `atlas_brain` imports. The
scaffold is still a staging boundary until the minimal runtime adapters are
hardened into customer-grade ports and the copied helper surface is trimmed to
the sellable workflows.


## Validation command

```bash
bash scripts/validate_extracted_content_pipeline.sh
```

This stable product entry point delegates to
`extracted/_shared/scripts/validate_extracted.sh extracted_content_pipeline`.

## ASCII compliance check

```bash
bash scripts/check_ascii_python.sh
```

This stable product entry point delegates to
`extracted/_shared/scripts/check_ascii_python.sh extracted_content_pipeline`.

## Import debt check

```bash
python scripts/check_extracted_imports.py
```

This stable product entry point delegates to
`extracted/_shared/scripts/check_extracted_imports.py extracted_content_pipeline`.
Known unresolved relative imports are tracked in `extracted_content_pipeline/import_debt_allowlist.txt`.

## Standalone readiness audit

```bash
python scripts/audit_extracted_standalone.py
python scripts/audit_extracted_standalone.py --fail-on-debt
```

The first command reports Atlas runtime coupling. The second is the product gate
for keeping the extracted package free of runtime `atlas_brain` imports.

## One-shot checks

```bash
bash scripts/run_extracted_pipeline_checks.sh
```

The one-shot runner enforces the standalone readiness audit with
`--fail-on-debt`; any new runtime `atlas_brain` import fails CI.

## Compatibility shims

To keep copied task modules importable inside this repo, package-level bridge modules are provided under `extracted_content_pipeline/` (for example `config.py`, `storage/database.py`, `pipelines/llm.py`, and `services/*`). Runtime imports no longer delegate to `atlas_brain`; most adapters are intentionally minimal local implementations.

B2B helper siblings required by `b2b_blog_post_generation.py` are also copied into `extracted_content_pipeline/autonomous/tasks/`.

These minimal adapters are extraction scaffolding. They need hardening before
shipping in the customer product.

The email/campaign generation slice is mapped in `docs/email_campaign_generation_pipeline.md`, with standalone productization requirements in `docs/standalone_productization.md`.

Reasoning is a host/product boundary, not copied into AI Content Ops. The
campaign generator consumes already-compressed reasoning through
`CampaignReasoningContextProvider`; see
`docs/reasoning_handoff_contract.md` for the accepted context shape and the
no-direct-import rule. Lightweight installs can use the packaged single-pass
provider; higher-depth installs can opt into the extracted reasoning core
through the multi-pass provider.

`campaign_opportunities.py` defines the customer-data contract for campaign
generation. Hosts can pass loose opportunity rows from a CRM, warehouse, or
vendor-intelligence feed; the product normalizes them into stable prompt and
draft metadata fields while preserving custom columns.

`campaign_customer_data.py` adds JSON/CSV file adapters and a
`FileIntelligenceRepository` so hosts can run the generator directly from
customer exports before wiring a database integration.

`campaign_source_adapters.py` adds a source-to-opportunity adapter for richer
review, transcript, sales-call, meeting, CRM deal/note, contract, renewal,
subscription, complaint, support-ticket, conversation, case, survey, NPS, CSAT,
or document rows. It preserves source text as campaign evidence and outputs the
same payload shape as the customer-data adapter.

## Campaign generation example

Run the standalone campaign generator against the included customer-data
payload:

```bash
python scripts/run_extracted_campaign_generation_example.py
```

Or run it against a customer JSON file and write drafts to disk:

```bash
python scripts/run_extracted_campaign_generation_example.py customer_payload.json --output campaign_drafts.json
```

CSV exports work too. The loader normalizes common CRM/warehouse columns such
as `company`, `vendor`, `email`, `title`, `pain_category`, `competitor`,
`opportunity_score`, and `urgency_score`, while preserving custom columns in
draft metadata:

```bash
python scripts/run_extracted_campaign_generation_example.py customer_opportunities.csv --format csv
```

Review, transcript, sales-call, meeting, CRM deal/note, contract, renewal,
subscription, complaint, support-ticket, conversation, case, survey, NPS, CSAT,
and document source rows can be converted into the same opportunity payload
first. Source rows can be JSON, JSONL, or CSV:

For a customer-owned support-ticket export, start with the packaged CSV or JSON
bundle examples. The CSV uses provider-style labels such as `Ticket ID`,
`Account Name`, `Vendor Name`, `Subject`, and `Description`; the JSON bundle
shows shared account metadata with multiple `support_tickets`:

```bash
python scripts/build_extracted_campaign_opportunities_from_sources.py \
  extracted_content_pipeline/examples/support_ticket_sources.csv \
  --format csv \
  --output support_ticket_opportunities.json

python scripts/build_extracted_campaign_opportunities_from_sources.py \
  extracted_content_pipeline/examples/support_ticket_bundle.json \
  --format json \
  --output support_ticket_bundle_opportunities.json
```

To produce a quick grounded FAQ artifact from those same support-ticket rows,
write Markdown directly:

```bash
python scripts/build_extracted_ticket_faq_markdown.py \
  extracted_content_pipeline/examples/support_ticket_sources.csv \
  --source-format csv \
  --window-days 90 \
  --support-contact "1-800-555-0100" \
  --require-output-checks \
  --output support_ticket_faq.md
```

The FAQ builder is deterministic and extractive: it groups ticket evidence by
pain point, writes article-style answers with numbered next steps, and cites
ticket quotes under each answer. The packaged support-ticket CSV is
intentionally small but repeated: it proves customer-worded headings, intent
condensation, action items, and source grounding. Add `--as-of-date
YYYY-MM-DD` with `--window-days` when you need a reproducible audit window
instead of today's date. Pass `--support-contact` when the published FAQ should
show a support phone number, email, or help URL; the builder never invents one.
Pass `--require-output-checks` in host smoke runs when weak FAQ output should
fail the command instead of producing a reviewable draft.

Reusable customer glossary and intent mappings can live in a JSON rule file:

```bash
python scripts/build_extracted_ticket_faq_markdown.py \
  extracted_content_pipeline/examples/support_ticket_sources.csv \
  --source-format csv \
  --documentation-term-file extracted_content_pipeline/examples/faq_documentation_terms.txt \
  --rule-file extracted_content_pipeline/examples/faq_custom_rules.json \
  --result-output support_ticket_faq_result.json \
  --output support_ticket_faq.md
```

The rule file accepts two optional arrays:

```json
{
  "intent_rules": [
    {"topic": "data freshness", "keywords": ["warehouse sync", "connector lag"]}
  ],
  "vocabulary_gap_rules": [
    ["SSO", "single sign-on"]
  ]
}
```

Intent rules group customer language under a product-specific FAQ topic.
Vocabulary-gap rules map customer terms to documentation terms passed with
`--documentation-term` or `--documentation-term-file`, so the result JSON and
Markdown can suggest alternate phrasing. Documentation-term files can be UTF-8
plain text, JSON, JSONL, or CSV. Text files use one term per line and ignore
blank or `#` comment lines. Structured files read common fields such as
`documentation_term`, `term`, `heading`, `title`, `page_title`, `name`, or
`label`. Repeat `--rule-file` to combine files. Explicit `--intent-rule` and
`--vocabulary-gap-rule` flags are placed before file rules, so command-line
overrides win when multiple rules match. Rule-file values use the same CLI
delimiter guardrails: intent topics cannot contain `=` or `,`, and keywords or
vocabulary aliases cannot contain `,`.

For large-upload validation, compare the scale smoke `run_summary.json` against
the checked-in failure profiles:

- `examples/faq_scale_density_limited_summary.json` shows a 1,000-row upload
  where most rows were skipped before FAQ generation. Fix source export fields,
  text columns, or filters before tuning the FAQ generator.
- `examples/faq_scale_output_check_failure_summary.json` shows healthy input
  density with failed FAQ output checks. In that case, investigate grouping,
  question wording, or action-step generation.

The first pass is `input_profile.usable_source_count` divided by
`input_profile.raw_row_count`. If that ratio is poor, treat the run as a source
prep problem. If the ratio is healthy and `failure.type` is `output_checks`,
debug the FAQ generator path. Use top-level `faq_run_summary` for the FAQ health
rollup: generated item count, weighted represented source volume, failed output
checks, warning count, and item score distribution.

The same artifact can run through the Content Ops execution seam by selecting
`faq_markdown` and passing inline `source_material`. It remains zero-provider.
When the host wires `PostgresTicketFAQRepository`, execution also persists the
Markdown document into `ticket_faq_markdown` and returns `saved_ids`.

To prove the persisted review lifecycle in one command, run the FAQ lifecycle
smoke. It generates from source rows, exports the saved draft, updates it to a
host-defined review status, and exports the reviewed row:

```bash
python scripts/smoke_content_ops_faq_lifecycle.py \
  extracted_content_pipeline/examples/support_ticket_sources.csv \
  --source-format csv \
  --account-id acct_123 \
  --review-status published \
  --json
```

If Atlas already has scraped B2B review rows, export one reliable source as
Content Ops source rows before generation. The G2 path is read-only, keeps only
canonical enriched reviews, and exports the negative/mixed quote-grade phrase
lane as `text` while preserving the full review for audit:

```bash
python scripts/export_content_ops_review_sources.py \
  --source-summary \
  --summary-sources g2,capterra,trustradius,trustpilot
```

```bash
python scripts/export_content_ops_review_sources.py \
  --source g2 \
  --vendor Slack \
  --limit 50 \
  --output g2_review_sources.jsonl
```

To prove the full review-source path in one command, run the operator smoke.
It checks readiness, exports quote-grade rows, validates source-row ingestion,
and generates offline drafts without a live LLM:

```bash
python scripts/smoke_content_ops_review_source_generation.py \
  --source g2 \
  --vendor Slack \
  --limit 2 \
  --default-field company_name="Acme Logistics" \
  --default-field contact_email=ops@example.com \
  --json
```

To prove the same source can run through the Postgres-backed path, use the
database smoke. It imports source rows into `campaign_opportunities` under the
provided account id, replaces matching imported opportunities by default, and
persists generated drafts. The default uses deterministic offline generation;
pass `--llm pipeline` to exercise the product `PipelineLLMClient`. If the
required Content Ops tables are not present, the smoke fails before import and
points at the migration runner:

```bash
python scripts/smoke_content_ops_review_source_postgres.py \
  --source g2 \
  --vendor Slack \
  --limit 1 \
  --account-id acct_123 \
  --default-field company_name="Acme Logistics" \
  --default-field contact_email=ops@example.com \
  --booking-url https://book.customer.com/demo \
  --llm pipeline \
  --json
```

For hosted Content Ops execution, pass the same selling asset through request
inputs so every generated campaign opportunity sees it in `opportunity_json`:

```json
{
  "outputs": ["email_campaign"],
  "inputs": {
    "target_account": "Acme Logistics",
    "offer": "Churn audit",
    "selling": {
      "booking_url": "https://book.customer.com/demo"
    }
  }
}
```

Public CFPB complaint narratives can also seed the same source-row path when
you want support-ticket-like evidence without using seller or review-site data.
The exporter reads CFPB's public CSV endpoint, keeps narrative-bearing
complaints, and emits `source_type="support_ticket"` rows. Account and contact
binding still comes from host defaults. The Postgres smoke defaults to offline
generation; pass `--llm pipeline` to use the configured product
`PipelineLLMClient`:

```bash
python scripts/export_content_ops_cfpb_sources.py \
  --company "Example Bank" \
  --search-term fees \
  --limit 25 \
  --output cfpb_sources.jsonl
```

To prove the public CFPB source can produce a grounded FAQ Markdown artifact
without a database or provider credentials, run the FAQ smoke. It fetches live
complaint narratives, converts them through the generic source-row adapter, and
fails if the FAQ output checks do not pass:

```bash
python scripts/smoke_content_ops_cfpb_faq_markdown.py \
  --search-term fees \
  --limit 3 \
  --support-contact "https://support.example.com" \
  --output-markdown cfpb_faq.md \
  --json
```

For a larger confidence run, raise both the requested source count and the scan
cap, and write every artifact to disk:

```bash
python scripts/smoke_content_ops_cfpb_faq_markdown.py \
  --search-term fees \
  --limit 1000 \
  --max-rows-scanned 5000 \
  --max-items 20 \
  --support-contact "https://support.example.com" \
  --output-source-rows cfpb_sources_1000.jsonl \
  --output-markdown cfpb_faq_1000.md \
  --json > cfpb_faq_1000.summary.json
```

The JSON summary includes `source_profile` so operators can separate source
prep from FAQ generation. Check `usable_source_count` against the requested
limit first. If it is low while `raw_row_count` is high, inspect
`missing_narrative_count`, `missing_complaint_id_count`, and
`skipped_row_count` before changing the generator. If
`stop_reason="max_rows_scanned"`, increase `--max-rows-scanned` or narrow the
CFPB filters. If `usable_source_count` is healthy but FAQ output checks fail,
investigate the FAQ grouping/output path.

```bash
python scripts/smoke_content_ops_cfpb_source_postgres.py \
  --company "Example Bank" \
  --search-term fees \
  --limit 1 \
  --account-id acct_123 \
  --default-field company_name="Acme Logistics" \
  --default-field contact_email=ops@example.com \
  --llm pipeline \
  --json
```

For customer-owned source files such as the packaged support-ticket CSV, use
the generic source-file Postgres smoke. It imports the file into
`campaign_opportunities`, runs offline draft generation through the Postgres
runner, and checks persisted draft target metadata:

```bash
python scripts/smoke_content_ops_source_file_postgres.py \
  extracted_content_pipeline/examples/support_ticket_sources.csv \
  --source-format csv \
  --account-id acct_123 \
  --json
```

Before converting or importing a customer export, inspect ingestion readiness:

```bash
python scripts/inspect_extracted_content_ingestion.py \
  g2_review_sources.jsonl \
  --source-rows \
  --source-format jsonl \
  --default-field company_name="Acme Logistics" \
  --default-field contact_email=ops@example.com \
  --json
```

```bash
python scripts/build_extracted_campaign_opportunities_from_sources.py \
  extracted_content_pipeline/examples/campaign_source_rows.jsonl \
  --output customer_opportunities.json

python scripts/build_extracted_campaign_opportunities_from_sources.py \
  extracted_content_pipeline/examples/campaign_source_bundle.json \
  --format json \
  --output customer_bundle_opportunities.json

python scripts/run_extracted_campaign_generation_example.py customer_opportunities.json
```

When multiple source-text fields are present, the adapter uses the first
recognized field in this order: `text`, `review_text`, `transcript`,
`content`, `body`, `quote`, `complaint`, `message`, `description`, `summary`,
`notes`, `feedback`, `feedback_text`, `response_text`, `comment_text`, then
`open_ended_response`. This precedence is global: a survey-shaped row with
`body` and `feedback` uses `body`. If none of those scalar fields are present,
it can build evidence text from nested `messages`, `comments`, `thread`,
`conversation`, or `entries` arrays. Within each nested message, message-shaped fields win first:
`text`, `message`, `body`, `content`, `comment`, `description`, `summary`,
then `notes`.

For quick offline previews, the generation CLI can consume those source rows
directly:

```bash
python scripts/run_extracted_campaign_generation_example.py \
  extracted_content_pipeline/examples/campaign_source_rows.jsonl \
  --source-rows \
  --source-format jsonl \
  --booking-url https://book.customer.com/demo

python scripts/run_extracted_campaign_generation_example.py \
  extracted_content_pipeline/examples/campaign_source_bundle.json \
  --source-rows \
  --source-format json

python scripts/run_extracted_campaign_generation_example.py \
  extracted_content_pipeline/examples/support_ticket_sources.csv \
  --source-rows \
  --source-format csv
```

They can also be loaded directly into the Postgres opportunity table:

```bash
python scripts/load_extracted_campaign_opportunities.py \
  extracted_content_pipeline/examples/campaign_source_rows.jsonl \
  --source-rows \
  --source-format jsonl \
  --account-id acct_123
```

For source-row CSV exports, pass `--source-format csv` to the same conversion,
generation, import, or smoke commands.
For anonymous review exports such as G2 rows without buyer/contact columns,
repeat `--default-field key=value` to bind the evidence to a target account at
conversion, generation, inspection, smoke, or import time. Source-row values
win when both the row and the default provide the same field. Use
`--booking-url` on the source conversion or generation commands to apply a
fallback `selling.booking_url` CTA to every converted opportunity.
For customer bundle JSON files that group collections such as `reviews`,
`support_tickets`, `surveys`, `calls`, `meetings`, `deals`, `account_notes`,
`contracts`, `renewals`, or `subscriptions` under shared account metadata, use
`--source-format json`; the packaged `campaign_source_bundle.json` demonstrates
that shape.
Rows with `recording_id` are treated as sales-call rows; hosts should rename
ambiguous screen-recording or webinar identifiers before import if they are not
sales-call evidence.
Rows with `deal_id` or `opportunity_id` are treated as CRM deal evidence; rows
with `note_id` or `activity_id` are treated as CRM note evidence.
Rows with `contract_id`, `renewal_id`, or `subscription_id` are treated as
contract, renewal, or subscription evidence.
Source-row keys are matched leniently for common export labels, so `Ticket ID`,
`Account Name`, `Pain Category`, and `Open Ended Response` normalize like
`ticket_id`, `account_name`, `pain_category`, and `open_ended_response`.

Generate cold-email and follow-up drafts for each opportunity by passing
channels:

```bash
python scripts/run_extracted_campaign_generation_example.py --channels email_cold,email_followup
```

Pass host-provided reasoning context without installing a reasoning engine:

```bash
python scripts/run_extracted_campaign_generation_example.py \
  --reasoning-context extracted_content_pipeline/examples/campaign_reasoning_context.json
```

`campaign_reasoning_data.FileCampaignReasoningContextProvider` matches context
rows by target id, company, email, or vendor, filters rows by `target_mode`
when provided, and feeds the normalized `CampaignReasoningContextProvider` port
documented in `docs/reasoning_handoff_contract.md`. Rows with blank
`target_mode` remain global fallbacks for lightweight legacy files.
For DB-backed installs, validate a loaded `campaign_reasoning_contexts` row
without running generation:

```bash
python scripts/check_extracted_campaign_reasoning_postgres.py \
  --account-id acct_123 \
  --target-id opp_123 \
  --company-name "Acme" \
  --json
```

To inventory or export durable reasoning rows before/after operator edits:

```bash
python scripts/list_extracted_campaign_reasoning_contexts.py \
  --account-id acct_123 \
  --target-mode vendor_retention \
  --selector "Acme" \
  --limit 100 \
  --format csv \
  --output reasoning-contexts.csv
```

The list/export CLI defaults to `--limit 20`; raise the limit deliberately for
larger inventories.

Hosts can also mount `api.reasoning_contexts` for `GET
/campaign-reasoning-contexts` inventory, `POST /campaign-reasoning-contexts`
single-row upsert, and scoped `DELETE /campaign-reasoning-contexts/{id}`. The
host injects pool, auth, and tenant scope; scoped
tenants cannot override `account_id` in the request body. Upsert requests must
use an explicit `context` object; alias keys and typoed context keys are
rejected. Delete requests require tenant scope or an explicit `account_id`
query parameter. Pass auth dependencies when mounting the router. Hosts can
also inject a `VisibilitySink` provider to emit metadata-only admin events for
reasoning context upserts and deletes.

To insert or update host-produced reasoning rows without hand-writing SQL, load
a JSON file with selectors plus a context payload:

```bash
python scripts/upsert_extracted_campaign_reasoning_contexts.py \
  reasoning-contexts.json \
  --account-id acct_123 \
  --target-mode vendor_retention \
  --selector "Acme" \
  --validate-opportunities \
  --audit-log logs/reasoning-context-upserts.jsonl \
  --dry-run
```

The input may be a single row, an array, or a wrapper such as
`{"contexts": [...]}`. Each row can provide `selectors`, selector fields such as
`target_id` / `company_name` / `contact_email`, and either `context`,
`reasoning_context`, or `campaign_reasoning_context`. Omit `--dry-run` after
validation to write the rows. `--audit-log` appends metadata-only JSONL entries
after successful writes. Add `--validate-opportunities` when the host wants the
CLI to confirm each context row matches an active `campaign_opportunities` row
before writing.

For lightweight installs that do not already have reasoning JSON, use
`services.single_pass_reasoning_provider.SinglePassCampaignReasoningProvider`.
It calls the configured `LLMClient` once per opportunity with the packaged
`digest/b2b_campaign_reasoning_context` prompt and returns the same normalized
context shape. This is not a multi-hop graph reasoner; it is the small packaged
Tier 1 path for "source row in, reasoned draft out."

The example CLI can wire that provider directly when the product LLM adapter is
configured:

```bash
python scripts/run_extracted_campaign_generation_example.py \
  --llm pipeline \
  --single-pass-reasoning
```

For multi-step reasoning backed by `extracted_reasoning_core`, use the
multi-pass provider:

```bash
python scripts/run_extracted_campaign_generation_example.py \
  --llm pipeline \
  --multi-pass-reasoning \
  --multi-pass-depth L3
```

Add `--quality-revalidation` to the offline example when you want the
standalone campaign specificity gate to reject drafts with placeholder tokens
or missing configured proof-term support before they are returned. When
enabled, the generator also adds normalized `campaign_proof_terms` from
reasoning anchors/witnesses/proof points to the prompt payload before the LLM
call. Rejected drafts return `quality_revalidation` details in
`CampaignGenerationResult.errors`, including blocking issues and unused proof
terms.

Use host-provided prompt contracts by pointing at a markdown skill directory:

```bash
python scripts/run_extracted_campaign_generation_example.py \
  --skills-root customer_skills
```

Custom prompts with the same skill name override packaged prompts; missing
prompts fall back to the bundled `skills/digest/*.md` files.

The example uses in-memory product ports and an offline deterministic LLM stand
in, so it does not need Atlas, a database, or provider credentials. It proves
the customer-data path: JSON opportunities in, normalized campaign drafts out.

For a one-command host smoke test that fails when no usable draft is produced:

```bash
python scripts/smoke_extracted_content_pipeline_host.py
```

The smoke command uses the packaged example payload, offline deterministic LLM,
and cold-email plus follow-up channels by default. Pass a customer JSON or CSV
file to validate a host export before wiring Postgres or send providers. Source
row exports can use the same smoke command with `--source-rows`:

```bash
python scripts/smoke_extracted_content_pipeline_host.py \
  extracted_content_pipeline/examples/campaign_source_rows.jsonl \
  --source-rows \
  --source-format jsonl

python scripts/smoke_extracted_content_pipeline_host.py \
  extracted_content_pipeline/examples/campaign_source_bundle.json \
  --source-rows \
  --source-format json
```

For source-row CSV exports, use `--source-format csv`.

To run the same example through the product LLM adapter, configure the
`EXTRACTED_CAMPAIGN_LLM_*` environment variables and pass `--llm pipeline`:

```bash
python scripts/run_extracted_campaign_generation_example.py --llm pipeline
```

For database-backed runs, apply the product migrations, set
`EXTRACTED_DATABASE_URL`, and run the Postgres generation runner. It reads
`campaign_opportunities`, generates drafts, and persists them into
`b2b_campaigns`:

```bash
python scripts/run_extracted_content_pipeline_migrations.py --dry-run
python scripts/run_extracted_content_pipeline_migrations.py
```

The migration command reads `EXTRACTED_DATABASE_URL` first, then `DATABASE_URL`.
Pass `--database-url` explicitly when a host app keeps product data in a
separate database.

Load customer opportunities from JSON or CSV into the product table:

```bash
python scripts/load_extracted_campaign_opportunities.py customer_opportunities.csv --dry-run
python scripts/load_extracted_campaign_opportunities.py customer_opportunities.csv --account-id acct_123 --replace-existing
```

The loader uses the same normalization contract as the offline example. It is
append-only by default; `--replace-existing` deletes matching target ids for
the selected account and target mode before inserting the new rows.

For the full database-backed host install path, see
`docs/host_install_runbook.md`.

Load host-produced blog blueprints into `blog_blueprints` before running the
blog-post output. The JSON file can be an array, a single object, or an object
with a `blueprints` array:

```bash
python scripts/load_extracted_blog_blueprints.py blog_blueprints.json \
  --account-id acct_123 \
  --target-mode vendor_retention \
  --dry-run

python scripts/load_extracted_blog_blueprints.py blog_blueprints.json \
  --account-id acct_123 \
  --target-mode vendor_retention
```

Before connecting workers to a customer database or provider, inspect local
install readiness without opening network or DB handles:

```bash
python scripts/check_extracted_content_install.py --profile generation --llm offline
python scripts/check_extracted_content_install.py --profile all --sender resend
python scripts/check_extracted_content_install.py --profile all --sender resend --json
```

```bash
python scripts/run_extracted_campaign_generation_postgres.py --account-id acct_123 --limit 10
```

The Postgres runner accepts the same channel expansion:

```bash
python scripts/run_extracted_campaign_generation_postgres.py --account-id acct_123 --channels email_cold,email_followup
```

It also accepts the same host-provided reasoning JSON as the offline example:

```bash
python scripts/run_extracted_campaign_generation_postgres.py \
  --account-id acct_123 \
  --reasoning-context extracted_content_pipeline/examples/campaign_reasoning_context.json
```

Add `--quality-revalidation` to the Postgres runner to run the same campaign
specificity gate before generated drafts are persisted. The same flag adds
normalized proof terms to the prompt payload so generated drafts have the
evidence terms the gate will later require. Failed drafts include compact
revalidation details in the runner result errors.

Or generate lightweight reasoning context during the DB-backed run:

```bash
python scripts/run_extracted_campaign_generation_postgres.py \
  --account-id acct_123 \
  --single-pass-reasoning
```

Or opt into extracted reasoning-core multi-pass context:

```bash
python scripts/run_extracted_campaign_generation_postgres.py \
  --account-id acct_123 \
  --multi-pass-reasoning \
  --multi-pass-depth L3
```

Use `--skills-root customer_skills` on the Postgres runner for the same
host-prompt override behavior.

Export generated drafts for review without writing SQL:

```bash
python scripts/export_extracted_campaign_drafts.py --account-id acct_123 --limit 20
python scripts/export_extracted_campaign_drafts.py --account-id acct_123 --format csv --output campaign_drafts.csv
```

The draft export keeps the full `metadata` JSON and also exposes scan-friendly
summary fields: `generation_input_tokens`, `generation_output_tokens`,
`generation_total_tokens`, `generation_parse_attempts`,
`reasoning_context_used`, `reasoning_wedge`, and `reasoning_confidence`.

After review, update selected draft rows without writing SQL:

```bash
python scripts/review_extracted_campaign_drafts.py <campaign-id> --account-id acct_123 --status approved
python scripts/review_extracted_campaign_drafts.py <campaign-id> --account-id acct_123 --status queued --from-email audit@customer.com
python scripts/review_extracted_campaign_drafts.py <campaign-id> --account-id acct_123 --status cancelled --reason "customer rejected"
```

Generated reports, landing pages, sales briefs, and FAQ Markdown documents use
the matching asset review CLI:

```bash
python scripts/review_extracted_content_assets.py --asset report --id <report-id> --account-id acct_123 --status approved
python scripts/review_extracted_content_assets.py --asset landing_page --id <landing-page-id> --account-id acct_123 --status queued
python scripts/review_extracted_content_assets.py --asset sales_brief --id <brief-id> --account-id acct_123 --status rejected
python scripts/review_extracted_content_assets.py --asset faq_markdown --id <faq-id> --account-id acct_123 --status approved
```

FastAPI hosts can mount the generated asset router for the same report, landing
page, sales brief, and FAQ Markdown review loop:

```python
from fastapi import Depends

from extracted_content_pipeline.api.generated_assets import create_generated_asset_router


app.include_router(
    create_generated_asset_router(
        pool_provider=get_pool,
        scope_provider=current_tenant_scope,
        dependencies=[Depends(require_content_ops_user)],
    )
)
```

This adds JSON draft listing, CSV/JSON export, and scoped status-update routes
for generated assets without importing Atlas API globals.

Hosts with FastAPI apps can mount the same draft review/export loop through a
router factory. The host injects its database pool, tenant scope, and auth
dependencies:

```python
from fastapi import Depends

from extracted_content_pipeline.api.b2b_campaigns import create_b2b_campaign_router


app.include_router(
    create_b2b_campaign_router(
        pool_provider=get_pool,
        scope_provider=current_tenant_scope,
        dependencies=[Depends(require_content_ops_user)],
    )
)
```

This adds JSON draft listing, CSV/JSON export, and approve/queue/cancel/expire
review routes without importing Atlas API globals.

Amazon seller installs can mount the seller-specific router. It adds seller
target CRUD, hosted category refresh and opportunity preparation triggers, plus
seller draft list/export/review routes locked to `target_mode="amazon_seller"`:

```python
from fastapi import Depends

from extracted_content_pipeline.api.seller_campaigns import create_seller_campaign_router


app.include_router(
    create_seller_campaign_router(
        pool_provider=get_pool,
        scope_provider=current_tenant_scope,
        dependencies=[Depends(require_content_ops_user)],
    )
)
```

Hosts can call `POST /seller/intelligence/refresh`,
`POST /seller/opportunities/prepare`, or
`POST /seller/operations/refresh-and-prepare` from an admin UI or scheduler.
Draft generation still runs through the worker/CLI path so hosts can control
LLM provider policy and runtime separately.

Before generating seller drafts, prepare seller opportunities from active
seller targets and cached category intelligence:

```bash
python scripts/prepare_extracted_seller_campaign_opportunities.py \
  --account-id acct_123 \
  --category supplements \
  --replace-existing

python scripts/run_extracted_campaign_generation_postgres.py \
  --account-id acct_123 \
  --target-mode amazon_seller \
  --channels email_cold,email_followup
```

Send queued drafts through the configured provider:

```bash
export EXTRACTED_RESEND_API_KEY="re_..."
python scripts/send_extracted_campaigns.py \
  --provider resend \
  --default-from-email audit@customer.com \
  --limit 10
```

Ingest Resend webhook payloads into the campaign tables:

```bash
export EXTRACTED_RESEND_WEBHOOK_SECRET="whsec_..."
python scripts/ingest_extracted_campaign_webhook.py \
  --body-file resend-event.json \
  --headers-json resend-headers.json \
  --json
```

Hosts with FastAPI apps can mount the same policy through a router factory:

```python
from extracted_content_pipeline.api.campaign_webhooks import (
    create_campaign_webhook_router,
)
from extracted_content_pipeline.campaign_send import verify_unsubscribe_token as verify_token


async def verify_unsubscribe_token(email: str, token: str) -> bool:
    return verify_token(
        email,
        token,
        get_unsubscribe_token_secret(),
    )


app.include_router(
    create_campaign_webhook_router(
        pool_provider=get_pool,
        signing_secret_provider=get_resend_webhook_secret,
        unsubscribe_token_verifier=verify_unsubscribe_token,
    )
)
```

The unsubscribe route accepts both `GET` and RFC 8058 one-click `POST`
requests. By default, hosts must provide an unsubscribe-token verifier so a
public query string cannot suppress arbitrary recipient addresses. Use the same
secret for `CampaignSendConfig.unsubscribe_token_secret` or
`EXTRACTED_CAMPAIGN_UNSUBSCRIBE_TOKEN_SECRET` so generated unsubscribe links
carry tokens the router can verify.

Refresh campaign analytics after send or webhook updates:

```bash
python scripts/refresh_extracted_campaign_analytics.py --json
```

Progress due sequences and queue generated follow-up drafts:

```bash
python scripts/progress_extracted_campaign_sequences.py \
  --from-email audit@customer.com \
  --limit 10

python scripts/progress_extracted_campaign_sequences.py \
  --llm offline \
  --json
```

For non-FastAPI worker installs, the four operational CLIs can append the same
start/completed/failed telemetry to a JSONL audit trail:

```bash
python scripts/run_extracted_campaign_generation_postgres.py \
  --account-id acct_123 \
  --visibility-jsonl /var/log/content-ops/campaign-events.jsonl

python scripts/send_extracted_campaigns.py \
  --provider resend \
  --default-from-email audit@customer.com \
  --visibility-jsonl /var/log/content-ops/campaign-events.jsonl

python scripts/read_extracted_campaign_visibility.py \
  /var/log/content-ops/campaign-events.jsonl \
  --operation send_queued \
  --limit 10
```

Hosts with FastAPI apps can mount draft generation, send, sequence progression,
and analytics worker triggers through a hosted operations router. The host
injects its database pool, sender, optional LLM/skill/reasoning providers, and
auth dependencies; request payloads only control tenant scope, target/channel,
filters, and batch sizing.

```python
from fastapi import Depends

from extracted_content_pipeline.api.campaign_operations import (
    CampaignOperationsApiConfig,
    create_campaign_operations_router,
)


app.include_router(
    create_campaign_operations_router(
        pool_provider=get_pool,
        sender_provider=get_campaign_sender,
        llm_provider=get_campaign_llm,
        skills_provider=get_campaign_skills,
        reasoning_context_provider=get_campaign_reasoning_context,
        config=CampaignOperationsApiConfig(
            send_default_from_email="audit@customer.com",
            sequence_from_email="audit@customer.com",
            generation_quality_revalidation=True,
        ),
        dependencies=[Depends(require_content_ops_admin)],
    )
)
```

If the host does not have a separate reasoning provider but does provide LLM and
skill providers, it can omit `reasoning_context_provider` and enable the
packaged single-pass provider:

```python
config=CampaignOperationsApiConfig(generation_single_pass_reasoning=True)
```

Hosts that want extracted reasoning-core orchestration instead can enable the
multi-pass provider:

```python
config=CampaignOperationsApiConfig(generation_multi_pass_reasoning=True)
```

This adds `POST /campaigns/operations/drafts/generate`,
`POST /campaigns/operations/send/queued`,
`POST /campaigns/operations/sequences/progress`, and
`POST /campaigns/operations/analytics/refresh` without exposing provider
credentials, sender identity, unsubscribe policy, or LLM configuration through
HTTP payloads.

It also adds `GET /campaigns/operations/status` for admin dashboards. The
status route reports database availability, injected provider presence, feature
readiness, and configured limits without resolving sender/LLM/skill providers
or exposing secrets.

For the multi-asset Content Ops control surface, run the offline execution
smoke before wiring real asset services:

```bash
python scripts/smoke_extracted_content_ops_execution.py
python scripts/smoke_extracted_content_ops_execution.py --outputs email_campaign,report --target-mode challenger_intel --no-quality-gates --json
python scripts/smoke_extracted_content_ops_execution.py --outputs email_campaign,landing_page --with-reasoning --json
python scripts/smoke_extracted_content_ops_execution.py --outputs email_campaign,landing_page --with-reasoning --reasoning-provider postgres-fixture --json
python scripts/smoke_extracted_content_ops_execution.py --outputs signal_extraction --source-vendor HubSpot --source-max-text-chars 400 --json
python scripts/smoke_extracted_content_ops_execution.py --outputs faq_markdown --source-type support_ticket --source-title "login reset" --json
python scripts/smoke_content_ops_faq_lifecycle.py --account-id acct_123 --review-status published --json
```

This validates the full campaign preset and the deterministic source-material
normalization and FAQ Markdown paths through host-injected services without opening database,
network, sender, or LLM handles. `--no-quality-gates` is an execution-smoke
override for checking the request wiring; production hosts should leave quality
gates enabled unless they intentionally disable them in their own policy layer.
`--with-reasoning` attaches the default `sample` host reasoning provider to
the fake generated-asset services. In reasoning mode, the smoke fails unless
the JSON output includes `result.reasoning_contexts_used`,
`reasoning.contexts_used`, and `reasoning.consumed_contexts` whenever the
usage count is positive. `--reasoning-provider postgres-fixture` switches that
offline provider to the real Postgres reasoning adapter backed by an in-memory
asyncpg-shaped pool, so hosts can validate the DB-adapter contract without a
live database.

Hosts can inject a `visibility_provider` when mounting the router. The four
POST operation routes emit best-effort `campaign_operation_started`,
`campaign_operation_completed`, and `campaign_operation_failed` events through
the `VisibilitySink` port so dashboards can show worker activity without the
content product owning a dashboard store. Sink failures are logged and do not
change operation responses.

For local dashboards or host audit logs, wire a product-owned visibility sink:

```python
from extracted_content_pipeline.campaign_visibility import JsonlVisibilitySink

visibility = JsonlVisibilitySink("/var/log/content-ops/campaign-events.jsonl")

app.include_router(
    create_campaign_operations_router(
        pool_provider=get_pool,
        sender_provider=get_campaign_sender,
        visibility_provider=lambda: visibility,
    )
)
```

Mount this router beside `create_b2b_campaign_router` to run the hosted B2B
flow without SQL in the admin UI:

1. `GET /campaigns/operations/status` lets the admin UI enable only ready
   operations.
2. `POST /campaigns/operations/drafts/generate` creates scoped draft rows from
   active `campaign_opportunities`.
3. `GET /b2b/campaigns/drafts` or `/drafts/export` lets operators inspect the
   generated drafts.
4. `POST /b2b/campaigns/drafts/review` moves selected drafts to `queued` after
   approval.
5. `POST /campaigns/operations/send/queued` sends approved queued drafts
   through the injected sender.
6. `POST /campaigns/operations/analytics/refresh` refreshes packaged funnel
   reporting after send/webhook activity.

## Import smoke test

```bash
python scripts/smoke_extracted_pipeline_imports.py
```

## Status tracker

Current extraction status is tracked in `extracted_content_pipeline/STATUS.md`.

## CI workflow

GitHub Actions workflow: `.github/workflows/extracted_pipeline_checks.yml` runs `bash scripts/run_extracted_pipeline_checks.sh` when extracted scaffold files change.

## File inventory

```bash
bash scripts/list_extracted_pipeline_files.sh
```

## LLM offline fallback

Set `EXTRACTED_PIPELINE_STANDALONE=1` to make the LLM bridge modules use their local no-op fallbacks instead of delegating to `extracted_llm_infrastructure`.

`campaign_llm_client.py` provides the product-owned `PipelineLLMClient`
adapter for campaign services. It satisfies the `campaign_ports.LLMClient`
port, resolves an LLM through the extracted LLM bridge when configured, and
normalizes `chat()` / `generate()` provider responses into `LLMResponse`.
`PipelineLLMClientConfig` and `create_pipeline_llm_client()` let a host wire
provider routing from explicit config, settings objects, or these environment
variables:

- `EXTRACTED_CAMPAIGN_LLM_WORKLOAD`
- `EXTRACTED_CAMPAIGN_LLM_PREFER_CLOUD`
- `EXTRACTED_CAMPAIGN_LLM_TRY_OPENROUTER`
- `EXTRACTED_CAMPAIGN_LLM_AUTO_ACTIVATE_OLLAMA`
- `EXTRACTED_CAMPAIGN_LLM_OPENROUTER_MODEL`

## Pipeline shims

`extracted_content_pipeline/pipelines/notify.py` provides a product-owned
notification dispatcher over the `VisibilitySink` port. It remains a safe no-op
when no sink is configured, and host apps can route emitted
`pipeline_notification` events to ntfy, Slack, dashboards, or audit streams.

Content-pipeline LLM bridge modules delegate to
`extracted_llm_infrastructure` instead of `atlas_brain`. That keeps the content
generation product boundary pointed at the extracted LLM/cost-optimization
product rather than at the monolith.

## Local utility shims

Several small utility shims provide product-owned local behavior by default so task imports do not require Atlas service modules:

- `config.py`: extracted settings from `settings.py`
- `pipelines/notify.py`: host-visible notification dispatcher backed by the
  `VisibilitySink` port
- `autonomous/tasks/_execution_progress.py`,
  `autonomous/tasks/_google_news.py`, `autonomous/tasks/_blog_ts.py`, and
  `autonomous/tasks/_blog_deploy.py`: product-owned utility helpers used by
  copied blog and campaign tasks
- `autonomous/tasks/_b2b_batch_utils.py`: product-owned Anthropic batch helper
  functions for metadata gates, request fingerprints, LLM slot resolution, and
  existing batch artifact reconciliation
- `autonomous/tasks/_blog_matching.py`: product-owned campaign-to-blog matcher
  with extracted base URL environment fallbacks
- `campaign_sequence_context.py` and
  `autonomous/tasks/_campaign_sequence_context.py`: product-owned sequence
  prompt/storage compaction helpers plus compatibility exports for copied tasks
- `autonomous/tasks/campaign_audit.py`: product-owned audit-log writer for
  campaign state changes
- `campaign_llm_client.py`: `PipelineLLMClient` adapter from the campaign
  `LLMClient` port to extracted LLM infrastructure services, with product-owned
  provider routing config
- `campaign_visibility.py`: reference in-memory and JSONL `VisibilitySink`
  adapters for hosted operations telemetry
- `storage/database.py` and `storage/models.py`: minimal `get_db_pool` and `ScheduledTask` fallbacks
- `campaign_postgres.py`: async Postgres adapters for intelligence,
  campaign, sequence, suppression, and audit ports, including the product-owned
  `campaign_opportunities` source table
- `campaign_postgres_generation.py`: product runner wiring
  `PostgresIntelligenceRepository`, `PostgresCampaignRepository`,
  `PipelineLLMClient`, and the local skill registry for DB-backed draft
  generation
- `campaign_postgres_export.py`: read-only draft export for host review flows
- `report_export.py`: read-only structured report export for host review flows
- `landing_page_export.py`: read-only landing page export for host review flows
- `sales_brief_export.py`: read-only sales brief export for host review flows
- `ticket_faq_export.py`: read-only ticket FAQ Markdown export for host review
  flows
- `scripts/export_extracted_content_assets.py`: host-facing report, landing
  page, sales brief, and FAQ Markdown export CLI
- `scripts/review_extracted_content_assets.py`: host-facing report, landing
  page, sales brief, and FAQ Markdown status-update CLI
- `campaign_postgres_seller_targets.py`: seller target CRUD/list helpers for
  Amazon seller campaign installs
- `campaign_postgres_seller_opportunities.py`: prepares Amazon seller
  `campaign_opportunities` rows from seller targets and cached category
  intelligence snapshots
- `campaign_postgres_seller_category_intelligence.py`: refreshes broad Amazon
  seller category snapshots from review and product metadata tables
- `campaign_postgres_send.py`: DB-backed queued send runner that composes the
  campaign, suppression, audit, and sender ports for host worker CLIs
- `campaign_postgres_analytics.py`: DB-backed analytics refresh runner that
  composes campaign and audit ports for host worker CLIs
- `campaign_postgres_webhooks.py`: DB-backed webhook ingestion runner that
  composes campaign, suppression, audit, and Resend verification ports for
  host worker CLIs
- `api/campaign_webhooks.py`: optional FastAPI router factory for host-mounted
  campaign webhook and unsubscribe routes
- `api/campaign_operations.py`: optional FastAPI router factory for
  host-mounted draft generation, send, sequence progression, and analytics
  operation triggers with optional `VisibilitySink` telemetry
- `api/b2b_campaigns.py`: optional FastAPI router factory for host-mounted
  B2B draft list/export/review routes
- `api/generated_assets.py`: optional FastAPI router factory for host-mounted
  report, landing page, and sales brief list/export/review routes
- `api/reasoning_contexts.py`: optional FastAPI router factory for
  host-mounted reasoning context list/upsert/delete routes with optional
  `VisibilitySink` audit events
- `api/seller_campaigns.py`: optional FastAPI router factory for host-mounted
  seller target management, category refresh, opportunity preparation, and
  seller draft review routes
- `campaign_postgres_sequence_progression.py`: DB-backed due-sequence worker
  that composes the sequence, audit, LLM, and skill ports for follow-up
  generation
- `campaign_postgres_import.py`: JSON/CSV customer opportunity import into the
  product `campaign_opportunities` table
- `storage/repositories/scheduled_task.py`: local execution metadata updater
- `skills/registry.py`: configurable markdown-backed skill registry
  implementing `.get()` and product `SkillStore.get_prompt()`, with optional
  host roots that override packaged prompt contracts
- `reasoning/archetypes.py`: product-owned deterministic churn-archetype scorer
  for extracted report builders
- `reasoning/temporal.py`: product-owned temporal analytics over vendor
  snapshot rows, including velocities, trends, category baselines, and
  anomaly serialization
- `reasoning/evidence_engine.py`: product-owned conclusion/suppression policy
  engine with built-in rules and optional host-provided evidence maps
- `services/__init__.py` and `services/protocols.py`: `llm_registry.get_active()` and `Message`
- `services/b2b/cache_runner.py`: local exact-cache request helpers and no-op lookup/store
- `services/b2b/enrichment_contract.py`: local enrichment contract fallbacks
- `services/scraping/sources.py`: `ReviewSource` enums and allowlist helpers
- `reasoning/wedge_registry.py`: `Wedge`, `get_wedge_meta`, and `validate_wedge`
- `services/blog_quality.py`: blog quality summary/revalidation helpers
- `services/company_normalization.py`: `normalize_company_name`
- `services/vendor_registry.py`: `resolve_vendor_name_cached`
- `services/apollo_company_overrides.py`, `services/b2b/corrections.py`, `services/tracing.py`, and `services/scraping/universal/html_cleaner.py`: local no-op or lightweight helpers

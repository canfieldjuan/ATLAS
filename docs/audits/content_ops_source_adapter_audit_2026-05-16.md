# Content Ops Source Adapter Audit

Created: 2026-05-16

## Verdict

Pause speculative source-shape additions. The adapter is useful and still
working, but the cumulative shape is now large enough that the next source
slice should be chosen by customer/export demand or by a small consolidation
step, not by adding another possible row family.

Update: the immediate consolidation follow-ups from this audit are complete.
PR #550 made source-type precedence explicit in code/tests, and PR #551 added
the per-row field lookup cache for provider-style aliases. Further source work
should now require a real host export or field-loss risk.

## Current Surface

The source adapter converts host source rows into the existing campaign
opportunity payload. It supports JSON, JSONL, CSV, nested source bundles, and
direct source-row ingestion through the campaign generation/import/smoke paths.

Current supported source families:

- Review and generic document rows.
- Transcripts, sales calls, meetings, and nested call/message turns.
- CRM deals, CRM notes, account notes, and activity records.
- Contracts, renewals, subscriptions, and commercial lifecycle notes.
- Complaints, support tickets, support cases, conversations, and nested ticket
  threads.
- Surveys, NPS, CSAT, feedback rows, and bundled collections.

Current size markers:

- 34 bundle collection keys.
- 21 source-id keys.
- 16 scalar source-text keys.
- 9 nested thread collection keys.
- 16 source-type inference branches.

## What Works

- The adapter keeps the host-facing contract stable: every source row becomes
  the same normalized campaign opportunity shape.
- Source text flows into `evidence`, preserving source id, source type, and
  source title.
- Exact-key inputs stay fast. Tolerant field aliases only run after exact key
  lookup misses.
- Parent account metadata flows into child source rows for bundled customer
  exports.
- Rows with no usable source text fail closed with a warning instead of
  producing generic drafts.

## Accumulated Risks

### Source-Type Precedence Is Explicit

The inference chain is ordered, and that order is part of the product contract.
PR #550 records that order as `_SOURCE_TYPE_PRECEDENCE` and locks the table in
tests. For example, review text wins over CRM ids, transcript text wins over
call ids, and renewal ids win over contract/subscription context.

Remaining risk: a future source family can still shift precedence, but that
now requires an explicit table/test update.

### Key Lists Are Becoming The Configuration Surface

Adding one source shape usually touches several lists: collection keys,
identifier keys, text keys, title keys, pain keys, and source-type inference.
That is still manageable, but it is no longer cheap enough to add every
plausible platform export proactively.

Risk: maintenance cost grows faster than product value if each hypothetical
export becomes a PR.

### Source Type Labels May Outrun Downstream Behavior

The generator currently receives source type as prompt context, but the
generation path does not yet select different prompts or policies per source
type. The labels are still useful for evidence review and future reasoning, but
not every new label creates immediate generation-quality lift.

Risk: adding more fine-grained labels can look like product progress without
changing output behavior.

### Alias Matching Is Broad By Design

Field matching now tolerates case, spacing, dashes, underscores, and compact
forms. This makes real CSV exports easier to ingest, but broad matching can
also accept typo-like labels.

Risk: unexpected host fields can silently map when a warning would be more
helpful.

Mitigation: PR #551 added a per-row lookup cache so broad matching does not
repeat normalized/compact scans for every helper call.

## Decision Rules

Add a new source family only when at least one condition is true:

- A customer or host export has that row family now.
- The row family carries evidence that cannot be represented by an existing
  generic field without losing important context.
- The row family unlocks a visible workflow, smoke, or import path that hosts
  can run immediately.

The slice owner should document the qualifying export, field-loss risk, or
workflow unlock in the PR plan. Reviewers should push back when a new source
shape is justified only by plausibility.

Do not add a new source family when:

- It is only a plausible future platform export.
- Existing `document`, `transcript`, `crm_note`, or `support_ticket` handling
  preserves the same evidence and metadata.
- The only difference is a provider-specific field name that the tolerant alias
  lookup already covers.

## Recommended Next Source Work

1. If a real host export is available, add only the minimal alias/source keys
   needed to load that file and include a fixture shaped like the export.
2. If no export is available, do not add more source breadth. The current
   consolidation items are complete; move to reasoning-policy depth instead.

## Performance Improvements

Provider-style field lookup now uses a per-row normalized/compact key cache.
Future performance work should be driven by measured ingestion bottlenecks, not
by source-shape breadth.

## Refactor Triggers

Move from the static precedence table and helper constants to a data-driven
registry when one of these happens:

- Source-type precedence grows beyond 20 table entries.
- A new source family requires edits to more than five independent constants.
- Two source families need custom precedence beyond a simple ordered table.
- Hosts need runtime extension of source families without product code changes.
- Downstream generation starts selecting prompts or quality policy by
  `source_type`.

## Testing Gaps

Current tests are strong at adapter level. The remaining gap is end-to-end
value testing: source row to opportunity to generated asset, with assertions
that the generated prompt or saved metadata actually uses the source-type
distinction.

This should be added when a real export fixture is available. Without that
fixture, broad end-to-end tests risk checking plumbing rather than product
quality. This is still a real coverage gap: until a fixture exists, source-type
distinctions could remain label-only without an output-quality regression test
catching it.

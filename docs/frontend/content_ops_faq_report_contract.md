# Content Ops FAQ Report Contract

This is the frontend/demo handoff for generated support-ticket FAQ reports.
The canonical FAQ producer is
`extracted_content_pipeline.ticket_faq_markdown.TicketFAQMarkdownResult.as_dict()`.
The canonical customer-facing deflection report producer is
`extracted_content_pipeline.faq_deflection_report.DeflectionReportArtifact.as_dict()`.

Use this contract for Content Ops `faq_markdown` execute results, persisted FAQ
detail hydration, Content Ops `faq_deflection_report` execute results, paid
deflection report artifacts, free deflection snapshots, and landing-page demos
that render the generated FAQ or deflection report artifact.

Do not use the compact search result row as the full report. Search returns a
ranked match preview; detail hydration returns the generated report.

## Generated Report

```ts
type TicketFAQMarkdownResult = {
  generated: number;
  markdown: string;
  items: TicketFAQItem[];
  source_count: number;
  ticket_source_count: number;
  // Tickets whose question appeared only once. They are excluded from FAQ
  // items and never billed as repeat work; a `non_repeat_tickets_excluded`
  // warning carries the same counts.
  non_repeat_ticket_count: number;
  non_repeat_question_count: number;
  output_checks: {
    uses_user_vocabulary: boolean;
    condensed: boolean;
    has_action_items: boolean;
  };
  warnings: Array<Record<string, unknown>>;
  saved_ids: string[];
};
```

## Deflection Report Artifact

`faq_deflection_report` execute steps return this shape in
`ContentOpsStepExecution.result`. The nested `faq_result` is the same generated
FAQ artifact described above; `markdown` is the customer-facing $1,500 report.

```ts
type FAQDeflectionReportArtifact = {
  markdown: string;
  summary: FAQDeflectionReportSummary;
  faq_result: TicketFAQMarkdownResult;
  report_model: DeflectionStructuredReport;
  evidence_export: DeflectionEvidenceExport;
};

type FAQDeflectionReportSummary = {
  generated: number;
  source_count: number;
  ticket_source_count: number;
  non_repeat_ticket_count: number;
  non_repeat_question_count: number;
  drafted_answer_count: number;
  no_proven_answer_count: number;
  output_checks: {
    uses_user_vocabulary: boolean;
    condensed: boolean;
    has_action_items: boolean;
  };
  top_question: string;
  top_opportunity_score: number;
};
```

The paid artifact also includes `report_model`, the first structured
`deflection.v1` section model. This is the forward-compatible contract for web,
PDF, email, Markdown, and export renderers. Current consumers may continue to
render `markdown`, but new surface-specific renderers should prefer
`report_model.sections` rather than parsing Markdown.

```ts
type DeflectionStructuredReport = {
  schema_version: "deflection.v1";
  title: string;
  summary: FAQDeflectionReportSummary;
  sections: DeflectionReportSection[];
};

type DeflectionReportSection = {
  id:
    | "support_tax"
    | "source_file"
    | "seo_targets"
    | "ranked_questions"
    | "priority_fix_queue"
    | "top_unresolved_repeats"
    | "drafted_resolutions"
    | "already_covered_still_recurring"
    | "backlog_table"
    | "outcome_diagnostics"
    | "question_details"
    | "complete_evidence"
    | string;
  title: string;
  priority: number;
  surfaces: Array<"web" | "pdf" | "email_summary" | "markdown" | "export" | string>;
  default_limit: number | null;
  required_data: string[];
  snapshot_safe_fields: string[];
  data: Record<string, unknown>;
};

type DeflectionReportProjectionContract = {
  schema_version: "deflection.v1";
  model_fields: ["schema_version", "title", "summary", "sections"];
  section_fields: [
    "id",
    "title",
    "priority",
    "surfaces",
    "default_limit",
    "required_data",
    "snapshot_safe_fields",
    "data"
  ];
  sections: DeflectionReportProjectionSection[];
};

type DeflectionReportProjectionSection = {
  id: string;
  title: string;
  priority: number;
  surfaces: string[];
  default_limit: number | null;
  required_data: string[];
  snapshot_safe_fields: string[];
  presence:
    | { mode: "required" }
    | { mode: "conditional"; condition: string };
  projected_fields: string[];
  optional_projected_fields?: string[];
  hosted_consumer_safe_fields: string[];
  collection?: {
    field: "rows" | "items" | "phrases" | string;
    item_type: "object" | "string" | string;
    projected_fields?: string[];
    hosted_consumer_safe_fields?: string[];
    nested_object_fields?: DeflectionReportProjectionNestedField[];
    nested_collection_fields?: DeflectionReportProjectionNestedCollection[];
  };
  nested_object_fields?: DeflectionReportProjectionNestedField[];
  nested_collection_fields?: DeflectionReportProjectionNestedCollection[];
  record_fields?: string[];
};

type DeflectionReportProjectionNestedField = {
  field: string;
  projected_fields: string[];
  hosted_consumer_safe_fields: string[];
};

type DeflectionReportProjectionNestedCollection = {
  field: string;
  item_type: "object" | "string" | string;
  projected_fields: string[];
  hosted_consumer_safe_fields: string[];
};

type DeflectionSnapshotProjectionContract = {
  schema_version: "deflection.v1";
  top_level_fields: [
    "summary",
    "top_questions",
    "locked_questions",
    "top_blind_spots",
    "teaser"
  ];
  fields: DeflectionSnapshotProjectionField[];
};

type DeflectionSnapshotProjectionField = {
  field: string;
  source_section: string;
  source_collection?: "rows" | "items";
  snapshot_safe_fields: string[];
  projected_fields: string[];
  full_answer_fields?: string[];
  preview_fields?: string[];
  policy?: "scoped_resolution_evidence_only";
  limit?: string;
};
```

Renderer rules:

- Sort sections by `priority`; do not rely on array position alone.
- Skip unknown section IDs or unsupported `surfaces` values rather than failing.
- Use `report_projection` as the paid structured-report source for future
  frontend type generation. `projected_fields` describes the full paid backend
  model; `hosted_consumer_safe_fields` is the smaller allowlist for URL/token
  hosted result-page payload construction.
- Use `presence.mode` before treating a section as required. `source_file`
  appears only when a source label exists, and `outcome_diagnostics` appears
  only when diagnostics render.
- Action section `items` include identity/delta and evidence fields in the full
  paid projection, but hosted result pages must allowlist-construct their item
  payload from `hosted_consumer_safe_fields` rather than validate-and-pass
  through raw rows.
- Treat nested arrays such as action-item `top_evidence` as
  `nested_collection_fields`; apply their item allowlist per row, not as a
  single nested object.
- `source_file.source_label` is not hosted-consumer-safe because it can contain
  a local/customer filename. Hosted pages should omit it unless a later slice
  normalizes it to a safe display label.
- Treat `snapshot_safe_fields` as the free Snapshot allowlist. A listed field
  must be safe to emit for every projected row in that section; paid answer
  bodies and steps are not snapshot-safe section fields. The only free answer
  body is the separately gated `snapshot.teaser.full_answer`.
- Treat the backend `snapshot_projection` contract as the source map from free
  Snapshot fields to paid report sections. In particular, `top_blind_spots` is
  sourced from `top_unresolved_repeats.items` and may project only `rank`,
  `question`, and `ticket_count`. Use this contract before changing frontend
  snapshot parsers; do not infer the projection from demo fixtures.
- Treat `complete_evidence` as export-only. It summarizes export size and should
  not be inlined into web/PDF surfaces or hosted result-page payloads.
- Breaking shape changes bump `schema_version`; additive sections should keep
  `schema_version: "deflection.v1"` and remain skippable by older renderers.

Paid artifacts generated by the current pipeline persist `report_model` in the
stored artifact JSON. Consumers reading historical paid artifacts must still
tolerate a missing or unsupported `report_model`: older records may predate the
structured model, and future schema versions must be handled by explicit
renderer support rather than guessed.

Paid consumers that only need the structured section contract should prefer
`GET /content-ops/deflection-reports/{request_id}/report-model`. It applies the
same tenant scope and paid lock as the full artifact route, returns only the
validated `deflection.v1` model, and returns 404 when a historical or drifted
paid artifact has no supported model.

The report Markdown always contains these customer-facing sections:

- `Support Tax Confirmation`
- `Your Help-Desk SEO Targeting List`
- `Ranked Question Opportunities`
- `Question Details and Evidence`

`Your Help-Desk SEO Targeting List` is a readability index, not the complete
evidence surface. Large reports render a deterministic top-N phrase list and an
omitted-count note. Complete source evidence is available in the paid artifact's
`evidence_export`; inline web/PDF renderers may stay bounded as long as they
link to or attach that export.

`Question Details and Evidence` is the canonical per-question detail pass. It
contains the answer status, publishable copy or no-proven-answer guidance,
vocabulary mappings, and representative source evidence for each ranked
question.

The action-oriented paid sections are a work queue, not a full ticket archive:

- `priority_fix_queue` carries enough ranked items for the largest advertised
  bounded surface (`pdf_limit`, currently 10) while `result_page_limit` tells
  the result page to render only the top three.
- Action rows include deterministic `priority_score` and bounded
  `priority_drivers` enum labels. The score is based on repeat volume,
  benchmark support cost, answer evidence status, confidence, reopened-ticket
  signal, and CSAT signal; it is not produced by an LLM or cloud classifier.
- Action rows also carry paid-only `repeat_key`, `cluster_id`,
  `identity_basis`, and `identity_confidence` fields. Use those for
  cross-run/monthly-delta matching; `rank` and evidence-export `question_id`
  are display/run-local identifiers and can change when priority changes.
- `top_unresolved_repeats` contains unresolved repeated questions only; one-off
  or low-confidence questions stay out of repeat accounting.
- `drafted_resolutions` contains publishable or adaptable answer drafts.
- `already_covered_still_recurring` flags proven answers whose status/CSAT
  signals suggest discoverability or answer-quality work.
- `backlog_table` is the broader bounded paid backlog; complete source evidence
  remains in `evidence_export`.

The paid artifact also includes a complete evidence export:

```ts
type DeflectionEvidenceExport = {
  schema_version: "deflection_evidence.v1";
  summary: {
    question_count: number;
    evidence_row_count: number;
    source_id_count: number;
    drafted_answer_count: number;
    no_proven_answer_count: number;
  };
  report_summary: FAQDeflectionReportSummary;
  questions: Array<{
    question_id: string;
    repeat_key: string;
    cluster_id: string;
    identity_basis:
  | "question_topic"
  | "question"
  | "source_ids"
      | "insufficient_identity";
    identity_confidence: "high" | "medium" | "low";
    rank: number;
    question: string;
    customer_wording: string;
    topic: string;
    ticket_count: number;
    weighted_frequency: number;
    opportunity_score: number;
    answer_evidence_status: "resolution_evidence" | "draft_needs_review" | string;
    resolution_evidence_scope: string;
    answer_linkage: "publishable_answer" | "needs_review";
    answer: string;
    steps: string[];
    source_ids: string[];
    evidence_quote_count: number;
    term_mappings: Array<Record<string, unknown>>;
    outcome_diagnostics: Record<string, unknown>;
  }>;
  evidence_rows: Array<{
    row_id: string;
    question_id: string;
    repeat_key: string;
    cluster_id: string;
    identity_basis: string;
    identity_confidence: "high" | "medium" | "low";
    rank: number;
    question: string;
    source_id: string;
    source_field: "evidence_quote" | "source_id" | string;
    evidence_quote: string;
    answer_evidence_status: string;
    resolution_evidence_scope: string;
    answer_linkage: "publishable_answer" | "needs_review";
  }>;
};
```

## Deflection Snapshot

The hosted free results page receives this shape before payment. It is the
canonical projection returned in `result.snapshot` from gated
`faq_deflection_report` execute responses and from
`GET /content-ops/deflection-reports/{request_id}/snapshot`.

```ts
type DeflectionSnapshot = {
  summary: {
    generated: number;
    drafted_answer_count: number;
    no_proven_answer_count: number;
    repeat_ticket_count: number;
    non_repeat_ticket_count: number;
    source_date_start?: string; // ISO date, only when source-date coverage is complete
    source_date_end?: string; // ISO date, only when source-date coverage is complete
    source_window_days?: number; // inclusive day count, only when coverage is complete
  };
  top_questions: DeflectionSnapshotQuestion[];
  locked_questions: DeflectionSnapshotLockedQuestion[];
  top_blind_spots: DeflectionSnapshotBlindSpot[];
  teaser: DeflectionSnapshotTeaser;
};

type DeflectionSnapshotQuestion = {
  rank: number;
  question: string;
  ticket_count: number;
  weighted_frequency: number;
  customer_wording: string;
};

type DeflectionSnapshotLockedQuestion = {
  rank: number;
  ticket_count: number;
};

type DeflectionSnapshotBlindSpot = {
  rank: number;
  question: string;
  ticket_count: number;
};

type DeflectionSnapshotTeaser = {
  full_answer: DeflectionSnapshotFullAnswer | null;
  previews: DeflectionSnapshotAnswerPreview[];
};

type DeflectionSnapshotFullAnswer = {
  rank: number;
  question: string;
  answer: string;
  steps: string[];
  answer_evidence_status: "resolution_evidence";
  resolution_evidence_scope: "scoped";
  weighted_frequency: number;
  source_count: number;
};

type DeflectionSnapshotAnswerPreview = {
  rank: number;
  question: string;
  answer_evidence_status: "resolution_evidence";
  resolution_evidence_scope: "scoped";
  weighted_frequency: number;
  step_count: number;
  source_count: number;
  body_withheld: true;
};
```

This shape intentionally excludes paid deliverable fields:

- no `markdown`
- no `faq_result`
- no answer text or `steps` outside `teaser.full_answer`
- no `evidence_quotes`
- no `source_ids`
- no vocabulary term mappings
- no locked question text; `locked_questions` contains only rank and raw
  ticket count
- no blind-spot paid action metadata; `top_blind_spots` contains only rank,
  question, and raw ticket count from unresolved repeated questions

The teaser is fail-closed: only scoped `resolution_evidence` FAQ items are
eligible. Preview entries never include answer body text.

`ticket_count` and `summary.repeat_ticket_count` are raw measured counts from
the report items/source rows. `repeat_ticket_count` sums only items whose
question was asked by at least two tickets; tickets behind single-ticket items
and excluded one-off questions are reported in
`summary.non_repeat_ticket_count` instead and must never feed repeat-work spend
copy. `weighted_frequency` is ranking metadata only and
must not feed spend or cost copy. If raw counts are unavailable, count-dependent
frontend projections should stay hidden rather than fall back to the ranking
score.

`summary.source_date_start`, `summary.source_date_end`, and
`summary.source_window_days` are optional and fail closed. ATLAS emits them only
when every contributing ticket source has a parseable date. If any of the three
fields is absent, the frontend must not normalize Support Tax copy to a source
window or infer a monthly pace from the upload.

## FAQ Item

```ts
type TicketFAQItem = {
  topic: string;
  question: string;
  question_source: "customer_wording" | "source_policy";
  summary: string;

  frequency: number;
  weighted_frequency: number;
  ticket_count: number;
  opportunity_score: number;
  failure_risk_score: number;
  failure_risk_signals: string[];

  answer: string;
  steps: string[];
  action_items: string[];
  answer_evidence_status: "resolution_evidence" | "draft_needs_review";
  resolution_source_count: number;
  when_to_contact_support: string;

  evidence_quotes: string[];
  source_ids: string[];
  source_labels: string[];
  source_type_counts: Record<string, number>;
  weighted_source_volume_by_type: Record<string, number>;
  source_date_span?: {
    start: string;
    end: string;
    window_days: number;
    dated_source_count: number;
    missing_source_count: number;
  };

  term_mappings: FAQTermMapping[];

  evidence_count: number;
  displayed_evidence_count: number;
};

type FAQTermMapping = {
  customer_term: string;
  documentation_term: string;
  suggestion: string;
  source_id_count: number;
  zero_result_source_count: number;
  failure_risk_score: number;
  failure_risk_signals: string[];
  opportunity_score: number;
  first_source_id: string;
};
```

## Persisted Detail

The detail route returns the persisted draft wrapper. Its `items` and `markdown`
fields are the same generated report artifact.

```ts
type TicketFAQDraft = {
  account_id: string;
  id: string;
  status: string;
  target_id: string;
  target_mode: string;
  title: string;
  markdown: string;
  items: TicketFAQItem[];
  source_count: number;
  ticket_source_count: number;
  output_checks: Record<string, boolean>;
  warnings: Array<Record<string, unknown>>;
  metadata: Record<string, unknown>;
};
```

## Compact Search Result

Search is intentionally smaller. Use `faq_id` from a selected result to hydrate
the full report detail.

```ts
type TicketFAQSearchResponse = {
  query: string;
  count: number;
  results: Array<{
    account_id: string;
    corpus_id: string;
    faq_id: string;
    target_id: string;
    target_mode: string;
    status: string;
    rank: number;
    topic: string;
    question: string;
    answer_summary: string;
    source_ids: string[];
    ticket_count: number;
    score: number;
  }>;
};
```

## Rendering Guidance

- Use `items` for ranked cards or sections, and `markdown` for the full report.
- For `faq_deflection_report`, render top-level `markdown` as the deliverable.
  Use `summary` for proof badges and `faq_result` for drill-down cards.
- For unpaid deflection results, render only `DeflectionSnapshot.summary`,
  `DeflectionSnapshot.top_questions`, `DeflectionSnapshot.locked_questions`,
  `DeflectionSnapshot.top_blind_spots`, and `DeflectionSnapshot.teaser`. Do not
  infer answer text, evidence, or source IDs from the snapshot.
- Show `source_count`, `ticket_source_count`, and `output_checks` as proof badges.
- Show `answer_evidence_status` near steps. `draft_needs_review` means the
  system found repeated customer wording but no uploaded resolution evidence, so
  the generated steps are review placeholders.
- In deflection reports, `drafted_answer_count` counts only FAQ opportunities
  backed by uploaded resolution evidence. `no_proven_answer_count` is not a
  failure; it is the list of repeated questions where support has not supplied a
  verified answer yet.
- Show `term_mappings` as vocabulary-gap suggestions. These are grounded in the
  supplied documentation terms/rules and customer wording.
- Treat `source_ids` and `evidence_quotes` as proof links/footnotes, not primary
  marketing copy.
- Treat `evidence_export` as the uncapped audit/detail surface. Hosted web and
  PDF renderers can cap inline evidence only when they preserve access to this
  export.

See [`content_ops_faq_report_example.json`](./content_ops_faq_report_example.json)
for a current compact FAQ example and
[`content_ops_faq_deflection_report_example.json`](./content_ops_faq_deflection_report_example.json)
for a current compact deflection report example. See
[`content_ops_faq_deflection_snapshot_example.json`](./content_ops_faq_deflection_snapshot_example.json)
for the free snapshot shape rendered before payment.

See
[`content_ops_faq_deflection_checkout_contract.md`](./content_ops_faq_deflection_checkout_contract.md)
for the Stripe Checkout metadata, paid-gate endpoints, and portfolio/ATLAS trust
boundary used to unlock the full report.

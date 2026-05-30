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
};

type FAQDeflectionReportSummary = {
  generated: number;
  source_count: number;
  ticket_source_count: number;
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

The report Markdown always contains these customer-facing sections:

- `Executive Summary`
- `Ranked Question Opportunities`
- `Drafted Answers With Proven Solutions`
- `No Proven Answer Yet`
- `Vocabulary Gaps`
- `Evidence Appendix`

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
  };
  top_questions: DeflectionSnapshotQuestion[];
};

type DeflectionSnapshotQuestion = {
  rank: number;
  question: string;
  weighted_frequency: number;
  customer_wording: string;
};
```

This shape intentionally excludes paid deliverable fields:

- no `markdown`
- no `faq_result`
- no answer text or `steps`
- no `evidence_quotes`
- no `source_ids`
- no vocabulary term mappings

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
- For unpaid deflection results, render only `DeflectionSnapshot.summary` and
  `DeflectionSnapshot.top_questions`. Do not infer answer text, evidence, or
  source IDs from the snapshot.
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
